import orjson
import aiohttp
import warnings

from urllib.parse import quote
from aiossechat import aiosseclient, SSEvent
from dataclasses import dataclass
from typing_extensions import Unpack
from typing import AsyncGenerator, TypeVar, TYPE_CHECKING

from .data_types import (CompletionInput, CompletionOutput, CompletionStreamOutput, LLMTool, ConditionProxy,
                         tidy_json_schema, EmbeddingInput, EmbeddingOutput, AIInput, EmbeddingModel)
from .common_utils.data_structs import BaseCondition

_T = TypeVar("_T")
warnings.simplefilter("once", UserWarning)

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
            if isinstance(model_filter, BaseCondition):
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

            payload['tools'] = tool_info    # type: ignore
        return payload  # type: ignore
    
    async def completion(self, /, **payload: Unpack[CompletionInput])->CompletionOutput:
        payload = self._validate_completion_input(**payload)
        return await self._request_ai(
            endpoint="/completion",
            payload=payload,    # type: ignore
            return_type=CompletionOutput,
        )
    
    async def stream_completion(self, /, **payload: Unpack[CompletionInput])->AsyncGenerator[CompletionStreamOutput, None]:
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