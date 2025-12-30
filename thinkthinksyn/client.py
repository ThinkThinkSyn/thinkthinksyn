import os
import re
import orjson
import aiohttp
import warnings
import logging

from urllib.parse import quote
from functools import partial
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined   # type: ignore
from pydantic.v1 import BaseModel as BaseModelV1
from dataclasses import dataclass
from types import UnionType
from typing_extensions import Unpack
from typing import (AsyncGenerator, TypeVar, TYPE_CHECKING, Sequence, TypeAlias, Any, overload, Literal,
                    Union, Callable)
from aiossechat import aiosseclient, SSEvent

from .data_types import (CompletionInput, CompletionOutput, CompletionStreamOutput, LLMTool, ConditionProxy,
                         tidy_json_schema, EmbeddingInput, EmbeddingOutput, AIInput, EmbeddingModel, ChatMsg,
                         ChatMsgWithRole, ChatMsgMedia, ChatMsgMedias, detect_media_type)
from .data_types.completion import _ChatMsgMediasList, _ChatMsgMedia

from .common_utils.data_structs import (BaseCondition, Image, Audio, Video, Condition)
from .common_utils.type_utils import (SerializableType, Empty, get_args, get_origin, check_type_is, get_pydantic_type_adapter)
from .common_utils.text_utils import json_repair_loads

warnings.simplefilter("once", UserWarning)

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_ST = TypeVar("_ST", bound=SerializableType)

_Media: TypeAlias = Image | Audio | Video
_PromptT: TypeAlias = str | _Media | ChatMsgMedia | ChatMsg | ChatMsgWithRole

def _tidy_single_media(media: _ChatMsgMedia)->ChatMsgMedia:
    if isinstance(media, (Audio, Image, Video)):
        media_type = 'audio' if isinstance(media, Audio) else ('image' if isinstance(media, Image) else 'video')
        return ChatMsgMedia(type=media_type, content=media.to_base64(url_scheme=True))
    elif isinstance(media, dict):
        media_type = detect_media_type(media.get('type', ''))
        common_keys = ('content', 'data', 'url', 'base64',)
        if media_type in ('image', 'audio', 'video'):
            if media_type == 'image':
                keys = common_keys + ('image', 'image_url', 'img', 'picture', 'image_data', 'photo')
            elif media_type == 'audio':
                keys = common_keys + ('audio', 'audio_url', 'sound', 'music', 'audio_data', 'voice')
            elif media_type == 'video':
                keys = common_keys + ('video', 'video_url', 'movie', 'clip', 'video_data')
            
            for k in keys:
                if (data := media.get(k, None)) is not None:
                    media.pop('type', None)
                    media.pop('content', None)
                    media.pop(k, None)
                    if isinstance(data, (Image, Audio, Video)):
                        data = data.to_base64(url_scheme=True)
                    return dict(type='image', content=data, **media)  # type: ignore
            raise ValueError(f"Cannot find image content in media dict. Expected keys: {keys}")
        else:
            raise ValueError(f"Unrecognized media type: {media.get('type', '')}")
        
    if isinstance(media, Path):
        media = str(media)
    if isinstance(media, str):
        # decide media type
        media_type, media_cls = None, None
        if media.startswith('data:'):
            media_type = media.split('/', 1)[0][5:]
            if media_type == 'image':
                media_cls = Image
            elif media_type == 'audio':
                media_cls = Audio
            elif media_type == 'video':
                media_cls = Video
            else:
                raise ValueError(f'Got invalid media type: {media_type}')
        else:
            if len(media) < 1024 and os.path.exists(media):
                suffix = media[-5:].split('.')[-1].lower()
                if (media_type:=detect_media_type(suffix)):
                    if media_type == 'image':
                        media_cls = Image
                    elif media_type == 'audio':
                        media_cls = Audio
                    elif media_type == 'video':
                        media_cls = Video
                    else:
                        raise ValueError(f'Got invalid media type: {media_type}')
                raise ValueError(f'Cannot determine media type from path: {media}')
            else:
                raise ValueError(f'Cannot load media from string: `{media[64:]}...`')
        return ChatMsgMedia(type=media_type, content=media_cls.Load(media))    # type: ignore
    raise TypeError(f"Invalid media type: {type(media)}")

def _tidy_msg_medias(medias: _ChatMsgMedia|_ChatMsgMediasList|ChatMsgMedias)->ChatMsgMedias:
    if isinstance(medias, (list, tuple)):
        return {i: _tidy_single_media(m) for i, m in enumerate(medias)}
    elif isinstance(medias, dict):
        first_key = next(iter(medias.keys()), None)
        if not isinstance(first_key, int):
            # single `ChatMsgMedia` dict
            return {0: _tidy_single_media(medias)}  # type: ignore
        return {k: _tidy_single_media(v) for k, v in medias.items()}    # type: ignore
    elif isinstance(medias, (Image, Audio, Video, str, Path)):
        return {0: _tidy_single_media(medias)}
    else:
        raise TypeError(f"Invalid medias type: {type(medias)}")

def _tidy_single_msg(msg: _PromptT, default_role='user')->ChatMsgWithRole:
    if isinstance(msg, str):
        return ChatMsgWithRole(role=default_role, content=msg)
    elif isinstance(msg, (Image, Audio, Video)):
        media_type = 'audio' if isinstance(msg, Audio) else ('image' if isinstance(msg, Image) else 'video')
        media = ChatMsgMedia(type=media_type, content=msg.to_base64(url_scheme=True))
        return ChatMsgWithRole(role=default_role, content='', medias={0: media})
    elif isinstance(msg, dict):
        if 'type' in msg:   # ChatMsgMedia
            return ChatMsgWithRole(role=default_role, content='', medias={0: _tidy_single_media(msg)})
        else:   # ChatMsg
            role = msg.pop('role', default_role)
            content = msg.pop('content', msg.pop('contents', ''))
            curr_medias = {}
            if isinstance(content, (Image, Audio)):
                content = ''
                curr_medias[0] = _tidy_single_media(content)
            elif isinstance(content, dict) and 'type' in content:
                content = ''
                curr_medias[0] = _tidy_single_media(content)
            if (medias:= msg.pop('medias', None)) is not None:
                extra_medias = _tidy_msg_medias(medias)    # type: ignore
                curr_medias.update(extra_medias)
            return ChatMsgWithRole(role=role, content=content, medias=curr_medias, **msg)  # type: ignore
    raise TypeError(f"Invalid prompt message type: {type(msg)}")

def _tidy_messages(prompt: _PromptT|Sequence[_PromptT], default_role='user')->list[ChatMsgWithRole]:
    if not isinstance(prompt, (list, tuple)):
        return _tidy_messages([prompt])   # type: ignore
    else:
        msgs = []
        last_role = default_role
        for p in prompt:
            msgs.append(_tidy_single_msg(p, default_role=last_role))
            last_role = msgs[-1]['role']
        return msgs

@dataclass
class ThinkThinkSyn:
    '''Client for interacting with the ThinkThinkSyn API.'''

    base_url: str = "https://api.thinkthinksyn.com/tts/ai"
    '''Client for interacting with the ThinkThinkSyn API.'''
    apikey: str = ""
    '''API key for authentication.'''

    # region basic utils
    def _ai_url(self, endpoint: str) -> str:
        return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    def _validate_ai_input(self, /, **payload: Unpack[AIInput]):
        model_filter = payload.pop('model_filter', None)
        if model_filter is not None:
            if isinstance(model_filter, (BaseCondition, Condition)):
                model_filter = str(model_filter)
            payload['model_filter'] = model_filter
        return payload
    
    async def _request_ai(self, endpoint:str, payload: dict, return_type: type[_T])->_T:
        endpoint = endpoint.lstrip("/")
        async with aiohttp.ClientSession() as session:
            if self.apikey:
                headers = {"Authorization": f"Bearer {self.apikey}"}
            else:
                headers = {}
            payload['stream'] = False
            async with session.post(self._ai_url(endpoint), json=payload, headers=headers) as response:
                response.raise_for_status()
                r = await response.json()
                if TYPE_CHECKING:
                    assert isinstance(r, return_type)
                return r

    async def _stream_request_ai(self, endpoint:str, payload: dict)->AsyncGenerator[SSEvent, None]:
        endpoint = endpoint.lstrip("/")
        if self.apikey:
            headers = {"Authorization": f"Bearer {self.apikey}"}
        else:
            headers = {}
        payload['stream'] = True
        async for e in aiosseclient(url=self._ai_url(endpoint), method='post', json=payload, headers=headers):
            yield e
    # endregion

    # region completion
    def _validate_completion_input(self, /, **payload: Unpack[CompletionInput])->CompletionInput:
        payload = self._validate_ai_input(**payload)  # type: ignore
        if (tool_info := payload.pop('tools', None)):
            tidied_tools = []
            if isinstance(tool_info, LLMTool):
                tidied_tools.append(tool_info.__dump__())
            elif isinstance(tool_info, (list, tuple)):
                for t in tool_info:
                    if isinstance(t, LLMTool):
                        t = t.__dump__()
                        if not isinstance(t, (str, dict)):
                            raise TypeError(f"Invalid tool info type: {type(t)}")
                        tidied_tools.append(t)
                    elif isinstance(t, (str, dict)):
                        tidied_tools.append(t)
                    else:
                        raise TypeError(f"Invalid tool info type: {type(t)}")
            elif isinstance(tool_info, (dict, str)):
                tidied_tools.append(tool_info)
            else:
                raise TypeError(f"Invalid tool info type: {type(tool_info)}")
            
            tool_info = []
            for t in tidied_tools:
                if isinstance(t, dict):
                    if 'params' in t and isinstance(t['params'], dict):
                        for k, v in t['params'].items():
                            t['params'][k] = tidy_json_schema(v)  # type: ignore
                    if 'return_type' in t and isinstance(t['return_type'], dict):
                        t['return_type'] = tidy_json_schema(t['return_type'])  # type: ignore
                tool_info.append(t)
            
            if 'history' in payload and 'messages' not in payload:
                payload['messages'] = payload.pop('history')  # type: ignore
            __tidy_messages__ = payload.pop('__tidy_messages__', True)
            if (msgs:=payload.pop('messages', None)) is not None:
                if __tidy_messages__:
                    msgs = _tidy_messages(msgs) # type: ignore
                payload['messages'] = msgs  # type: ignore

            payload['tools'] = tool_info    # type: ignore
        return payload  # type: ignore
    
    async def completion(self, /, **payload: Unpack[CompletionInput])->CompletionOutput:
        '''
        Complete the given prompt.
        For supported params, please refer to `thinkthinksyn.data_types.CompletionInput`.
        '''
        payload = self._validate_completion_input(**payload)
        return await self._request_ai(
            endpoint="/completion",
            payload=payload,    # type: ignore
            return_type=CompletionOutput,
        )
    
    async def stream_completion(self, /, **payload: Unpack[CompletionInput])->AsyncGenerator[CompletionStreamOutput, None]:
        '''
        Stream completion for the given prompt.
        For supported params, please refer to `thinkthinksyn.data_types.CompletionInput`.
        '''
        payload = self._validate_completion_input(**payload)
        async for event in self._stream_request_ai(
            endpoint="/completion",
            payload=payload,    # type: ignore
        ):
            if (data := event.data):
                if event.event == 'message':
                    yield {'event': 'message', 'data': event.data}
                else:
                    yield {'event': event.event, 'data': orjson.loads(data)}  # type: ignore
    
    @overload
    async def json_complete(
        self,
        prompt: _PromptT|Sequence[_PromptT],
        /,
        return_type: type[_ST],
        prompt_type: Literal["system", "user"] = "system",
        default: _T = None,
        retry: bool | int = 1,
        **kwargs: Unpack[CompletionInput],
    ) -> _ST|_T: ...
    
    @overload
    async def json_complete(
        self,
        prompt: _PromptT|Sequence[_PromptT],
        /,
        return_type: type[_ST],
        return_raw_output: Literal[False],
        prompt_type: Literal["system", "user"] = "system",
        default: _T = None,
        retry: bool | int = 1,
        **kwargs: Unpack[CompletionInput],
    ) -> _ST|_T: ...
    
    @overload
    async def json_complete(
        self,
        prompt: _PromptT|Sequence[_PromptT],
        /,
        return_type: type[_ST],
        return_raw_output: Literal[True],
        prompt_type: Literal["system", "user"] = "system",
        default: _T = None,
        retry: bool | int = 1,
        **kwargs: Unpack[CompletionInput],
    ) -> tuple[_ST|_T, CompletionOutput]: ...
    
    async def json_complete(    # type: ignore
        self,
        prompt: _PromptT|Sequence[_PromptT],
        /,
        return_type: type,
        return_raw_output: bool = False,
        prompt_type: Literal["system", "user"] = "system",
        default: Any = None,
        retry: bool | int = 1,
        **kwargs: Unpack[CompletionInput],
    ):
        '''
        Ask LLM to response in json format, and parse the result into the given `return_type`.
        
        Args:
            prompt: The prompt to complete. It can be:
                - string / Image / Audio / ChatMsgMedia / ChatMsg / ChatMsgWithRole
                - list of the above types
            return_type: The expected return type. It should be serializable, i.e.:
                - basic types, e.g. str, int, float, bool, list, dict
                - Pydantic BaseModel
                - classes whom has defined `__get_pydantic_core_schema__` method
            return_raw_output: Whether to return the raw completion output along with the parsed result.
            default: Default value to return if parsing FAILS(i.e. LLM fail to respond in given format).
            retry: Whether to retry when parsing fails. If an integer is provided, it indicates the maximum number of retries.
            **kwargs: Other kwargs for `completion` method.
        '''
        origin_kwargs = kwargs.copy()
        assert prompt, "`prompt` cannot be empty."
        if (msgs:= kwargs.pop('messages', None)) is not None:
            msgs = list(msgs) + _tidy_messages(prompt, default_role=prompt_type)  # type: ignore
        else:
            msgs = _tidy_messages(prompt, default_role=prompt_type)  # type: ignore
        assert msgs, "No prompt is given."
        
        if (schema := kwargs.pop('json_schema', None)) is not None:
            warnings.warn('`json_schema` parameter is provided in `json_complete` method, which will be overridden.', category=UserWarning)
        retry_count = int(retry)
        need_convert_to_tuple = False
        try:
            type_args = get_args(return_type)
        except:
            type_args = []
        try:
            type_origin = get_origin(return_type)
        except:
            type_origin = None

        is_basemodel_v2 = check_type_is(return_type, BaseModel) and (type_origin not in (Union, UnionType))
        is_basemodel_v1 = (not is_basemodel_v2 and check_type_is(return_type, BaseModelV1) and (type_origin not in (Union, UnionType)))
        if return_type == tuple or type_origin == tuple:
            if not type_args:
                return_type = type_origin = list
                need_convert_to_tuple = True
            elif len(type_args) == 2 and type_args[0] != Ellipsis and type_args[1] == Ellipsis:
                # e.g. tuple[int, ...]
                return_type = type_origin = list[type_args[0]]
                type_args = [type_args[0]]
                need_convert_to_tuple = True

        try_validate_types = [(return_type, None),]  # [(try_validate_type, callback),]
        if (return_type == list or type_origin == list) and len(type_args) == 1:
            # sometimes, for types like `list[SomeModel]`,  LLM will just return 1 single `SomeModel` object
            # instead of a list of `SomeModel` object. In this case, we will keep trying to validate the inner
            # type, and if success, we will wrap it into a list.
            try_validate_types.append((type_args[0], list))  # type: ignore

        def try_extract(
            s: str,
            try_validate_types: list[tuple[type, Callable[[Any], Any] | None]]
        ):
            if isinstance(s, str):
                try:
                    s = json_repair_loads(s)
                except:
                    ...
            for i, (try_type, callback) in enumerate(try_validate_types):
                try:
                    if is_basemodel_v2 or is_basemodel_v1:
                        model_fields = try_type.model_fields if is_basemodel_v2 else try_type.__fields__
                        required_fields = set()
                        for n, f in model_fields.items():
                            if f.default != PydanticUndefined and f.default_factory not in (None, PydanticUndefined):
                                required_fields.add(n)

                        def _check_dict(d: dict):
                            required = required_fields.copy()
                            for k in d:
                                if k in required:
                                    required.remove(k)
                            return len(required) == 0

                        if isinstance(s, dict):
                            if "properties" in s and "title" in s and "type" in s:
                                if any((k not in model_fields) for k in ["properties", "title", "type"]):
                                    properties, title, type_ = s["properties"], s["title"], s["type"]   # type: ignore
                                    cls_name = try_type.__name__.split(".")[-1]
                                    if title == cls_name and type_ == "object" and isinstance(properties, dict):
                                        s = properties
                                elif (
                                    len(s) == 1
                                    and (list(s.keys())[0] not in model_fields)
                                    and isinstance(s[list(s.keys())[0]], dict)
                                ):
                                    data = s[list(s.keys())[0]]
                                    if _check_dict(data):   # type: ignore
                                        s = data

                        if is_basemodel_v2:
                            val = try_type.model_validate(s)
                        else:
                            val = try_type.parse_obj(s)
                    else:
                        type_adapter = get_pydantic_type_adapter(try_type)
                        val = type_adapter.validate_python(s)
                    if callback:
                        val = callback(val)
                    return val
                except Exception as e:
                    if i != len(try_validate_types) - 1:
                        _logger.debug(f"Failed to validate the json response ```{s}``` to type {try_type}. Error: {type(e).__name__}:{e}. Trying next type: {try_validate_types[i+1][0]}" )
                        continue
                    else:
                        _logger.debug(f"Failed to validate the json response ```{s}``` to type {try_type}. Error: {type(e).__name__}:{e}")
                        return Empty
            return Empty
        extract = partial(try_extract, try_validate_types=try_validate_types)  # type: ignore
        if is_basemodel_v2:
            schema = return_type.model_json_schema()  # type: ignore
        elif is_basemodel_v1:
            schema = return_type.schema()
        else:
            type_adapter = get_pydantic_type_adapter(return_type)
            schema = type_adapter.json_schema()
        
        last_prompt: str = msgs[-1]['content']  # type: ignore
        last_prompt += "\n\nNOTE: Your response should follows the json schema below:"
        last_prompt += f"\n```\n{schema}\n```"
        last_prompt += f"\nReturn the valid json response only, without any other text."
        last_prompt += " The json response should have no indentation, meaning no newline characters and whitespace."
        msgs[-1]['content'] = last_prompt  # type: ignore
        
        payload = {
            '__tidy_messages__': False,     # no need to tidy again
            'messages': msgs,
            'json_schema': schema,
            **kwargs,
        }
        
        async def retry_or_raise(e, retry_count: int):
            if retry_count:
                retry_count -= 1
                _logger.debug(f"Failed to get json response from LLM. Error: {type(e).__name__}:{e}. Retrying..., {retry_count} retries left.")
                return await self.json_complete(
                    prompt,
                    return_type=return_type,
                    prompt_type=prompt_type,
                    default=default,
                    return_raw_output=return_raw_output,  # type: ignore
                    retry=retry_count,
                    **origin_kwargs,
                )  # type: ignore
            else:
                _logger.debug(f"Failed to get json response from LLM. Error: {type(e).__name__}:{e}.")
            if return_raw_output:
                raise ValueError(f"Failed to get json response from LLM. Error: {type(e).__name__}:{e}") from e
            return default  # type: ignore
        
        try:
            r: CompletionOutput = await self.completion(**payload)  # type: ignore
        except aiohttp.ClientResponseError as e:
            if e.status in (401, 403):
                raise PermissionError("Authentication failed. Please check your API key.") from e
            return await retry_or_raise(e, retry_count)
        except Exception as e:
            return await retry_or_raise(e, retry_count)
        json_r = Empty

        if check_type_is(return_type, (list, tuple)):
            try_patterns = [r"(\[.*\])", r"(\{.*\})"]  # try both `[...]` and `{...}`
        else:
            try_patterns = [r"(\{.*\})"]  # try `{...}` only

        while (json_r is Empty) and try_patterns:
            p = try_patterns.pop(0)
            if json_text := re.search(p, r['text'], re.DOTALL | re.MULTILINE):
                json_text = json_text.group(1)
                json_r = extract(json_text)     # type: ignore

        if json_r is Empty:
            if retry_count:
                retry_count -= 1
                _logger.debug(
                    f"{r['input']['model']} Failed to extract json response from {r['text']} to type {return_type}. Retrying..., {retry_count} retries left."
                )
                return await self.json_complete(
                    prompt,
                    return_type=return_type,
                    prompt_type=prompt_type,
                    default=default,
                    return_raw_output=return_raw_output,  # type: ignore
                    retry=retry_count,
                    **origin_kwargs,
                )  # type: ignore
            else:
                _logger.debug(f"{r['input']['model']} Failed to extract json response from {r['text']} to type {return_type}.")
            if return_raw_output:
                return default, r  # type: ignore
            return default  # type: ignore
        else:
            if need_convert_to_tuple and isinstance(json_r, Sequence) and not isinstance(json_r, str):
                json_r = tuple(json_r)
            if return_raw_output:
                return json_r, r  # type: ignore
            return json_r  # type: ignore
    # endregion
    
    # region embedding
    async def embedding(self, /, model: EmbeddingModel|str|None=None, **payload: Unpack[EmbeddingInput])->EmbeddingOutput:
        '''
        Get embedding for the given text.
        NOTE: if `model` or `model_filter` is not provided in the input, a default model will be selected,
             which may not be an optimal choice. It is recommended to provide at least one of them. 
        '''
        final_selected_model: str|None = None
        model_filter = payload.get('model_filter', None)
        if model is not None:
            if isinstance(model, EmbeddingModel):
                final_selected_model = model.Name   # type: ignore
                if isinstance(final_selected_model, ConditionProxy):
                    final_selected_model = None
        if not final_selected_model and model_filter is not None:
            if isinstance(model_filter, BaseCondition):
                for subcls in EmbeddingModel.__subclasses__():
                    if model_filter.validate(subcls, fuzzy=True):
                        final_selected_model = subcls.Name   # type: ignore
                        break
        if not final_selected_model:
            default = EmbeddingModel.Default()
            final_selected_model = default.Name
            if '/' in final_selected_model:     # to avoid url path issues
                if (alias := default.Alias):
                    final_selected_model = alias[0]

            warnings.warn(
                "No embedding model specified via `model` or `model_filter`. "
                f"Using default model `{final_selected_model}`. "
                "It is recommended to specify at least one of them for better results.",
                category=UserWarning,
            )
        
        return await self._request_ai(
            endpoint=f"/embedding/{quote(final_selected_model)}",
            payload=payload,    # type: ignore
            return_type=EmbeddingOutput,
        )
    # endregion
    

    
__all__ = ["ThinkThinkSyn"]