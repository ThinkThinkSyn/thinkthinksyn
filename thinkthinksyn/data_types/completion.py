import os
import re
import json
import html
import time
import asyncio
import logging

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from pydantic import BaseModel
from typing_extensions import TypeAliasType, NotRequired, Required, TypeIs
from typing import (Any, TypedDict, Literal, TypeAlias, Sequence, get_args, Iterable, Self, 
                    Callable, Awaitable, TYPE_CHECKING)

from ..common_utils.data_structs import Image, Audio, Video, PDF
from ..common_utils.concurrent_utils import SyncOrAsyncFunc, run_any_func, async_run_any_func
from ..common_utils.type_utils import create_pydantic_core_schema, get_pydantic_type_adapter
from ..common_utils.text_utils import full_width_text_tidy_up

from .base import AIInput, AIOutput, JsonSchema, current_thinkthinksyn_client
from .llm_tools import LLMTool
from .models import (LLM, LLM_DEFAULT_USER_ROLE_NAME, LLM_DEFAULT_ASSIST_ROLE_NAME, LLM_DEFAULT_SYSTEM_ROLE_NAME, LLM_DEFAULT_SYSTEM_ROLE_ALIAS,
                     _CommonLLM)

if TYPE_CHECKING:
    from ..client import ThinkThinkSyn

_logger = logging.getLogger(__name__)

ChatMsgMediaType: TypeAlias = Literal["image", "audio", "video"]
_ChatMsgMediaImageTypeAliases: TypeAlias = Literal["img", "image_url", 'picture', 'image_data', 'photo']
_ChatMsgMediaAudioTypeAliases: TypeAlias = Literal["sound", "music", 'audio_url', 'voice', 'audio_data']
_ChatMsgMediaVideoTypeAliases: TypeAlias = Literal["movie", "clip", 'video_url', 'video_data']
_ChatMsgMediaPDFTypeAliases: TypeAlias = Literal["pdf", "pdf_url", "pdf_data"]

class ChatMsgMedia(TypedDict, total=False):
    """
    A media content in the chat msg.
    NOTE:
        - each media is associated with a tag `<__MEDIA_{idx}__>` (or <__media__> <__MEDIA__>... is also acceptable) in the content.
        - 1 msg can have multiple media contents, e.g. image, audio, ...
    """

    type: ChatMsgMediaType | _ChatMsgMediaImageTypeAliases | _ChatMsgMediaAudioTypeAliases | _ChatMsgMediaVideoTypeAliases | _ChatMsgMediaPDFTypeAliases
    """type of the media content type"""
    content: str | Image | Audio | Video | PDF
    """Raw content of the media msg. It could be a URL/ file path/ base64, ...."""
    textified_content: str|None
    """
    The textified content, i.e. image/audio is translated into meaningful text for
    inputting to non-multimodal LLM.
    
    This value will only be set when `get_text_content` is called. 
    WARNING: You should not pass this value directly except you know what you are doing.
    """
    # type specific configs
    use_audio_in_video: bool
    '''whether to use audio when extracting meaning from video.
    This field is only valid when `type` is `video`.'''
    
class _TidiedChatMsgMedia(ChatMsgMedia):
    type: ChatMsgMediaType
    content: str | Image | Audio | Video | PDF
    textified_content: NotRequired[str|None]
    use_audio_in_video: NotRequired[bool]
    
def _try_get_from_dict(keys: Sequence[str], d: dict[str, Any]):
    for k in keys:
        if k in d:
            return d[k], k
    return None, None

def dump_chat_msg_media(media: "_ChatMsgMedia|_TidiedChatMsgMedia")->dict[str, Any]:
    if isinstance(media, (Image, Audio, Video, PDF, str, Path)):
        if isinstance(media, Path):
            media = str(media)
        media = {'content': media}
    if not isinstance(media, dict):
        raise ValueError("media should be ChatMsgMedia or dict-like object.")
    
    type = media.get('type', None)
    content_type = None
    content = media.get('content', media.get('data', None))
    if not content:
        content, key = _try_get_from_dict(
            (get_args(ChatMsgMediaType) +
            get_args(_ChatMsgMediaImageTypeAliases) +
            get_args(_ChatMsgMediaAudioTypeAliases) +
            get_args(_ChatMsgMediaVideoTypeAliases) +
            get_args(_ChatMsgMediaPDFTypeAliases)),
            d=media     
        )
        if key and not type:
            type = key
    
    if isinstance(content, Image):
        content_type = "image"
        adapter = get_pydantic_type_adapter(Image)
        content = adapter.dump_python(content)
    elif isinstance(content, Audio):
        content_type = "audio"
        adapter = get_pydantic_type_adapter(Audio)
        content = adapter.dump_python(content)
    elif isinstance(content, Video):
        content_type = "video"
        adapter = get_pydantic_type_adapter(Video)
        content = adapter.dump_python(content)
    elif isinstance(content, PDF):
        content_type = "pdf"
        raise NotImplementedError("Dumping PDF content is not implemented yet.")
    elif isinstance(content, str):
        if content.startswith('data:'):
            # base64 content
            prefix = content[5:].split(';')[0].lower()
            if 'image/' in prefix:
                content_type = "image"
            elif 'audio/' in prefix:
                content_type = "audio"
            elif 'video/' in prefix:
                content_type = "video"
            elif 'application/pdf' in prefix:
                content_type = "pdf"
    
    type = type or content_type
    if type in get_args(_ChatMsgMediaImageTypeAliases):
        type = "image"
    elif type in get_args(_ChatMsgMediaAudioTypeAliases):
        type = "audio"
    elif type in get_args(_ChatMsgMediaVideoTypeAliases):
        type = "video"
    elif type in get_args(_ChatMsgMediaPDFTypeAliases):
        type = "pdf"
    
    if not type:
        raise ValueError("Cannot detect media type automatically. Please specify `type` field in ChatMsgMedia.")
    return {
        'type': type,
        'content': content,
        'textified_content': media.get('textified_content', None),
        'use_audio_in_video': media.get('use_audio_in_video', False),
    }

def is_chat_msg_media(obj: Any)->TypeIs[ChatMsgMedia]:
    '''Check whether the given object is a ChatMsgMedia.'''
    if not isinstance(obj, dict):
        return False
    if 'type' not in obj:
        if 'content' not in obj:
            return False
        content = obj['content']
        if not isinstance(content, (str, Image, Audio, Video, PDF)):
            return False
    return True

def create_chat_msg_media_from_media(media: Image|Audio|Video|PDF)->ChatMsgMedia:
    '''Create ChatMsgMedia from given media object.'''
    if not isinstance(media, (Image, Audio, Video, PDF)):
        raise ValueError(f'Invalid media object: {media}. Expected Image, Audio, Video or PDF.')
    type = type(media).__name__.lower()
    return {
        'type': type,
        'content': media,
        'textified_content': None,
    }

class _OpenAIFormatImageInnerT1(TypedDict):
    url: str
class _OpenAIFormatImageInnerT2(TypedDict):
    image_url: str
class _OpenAIFormatImageT1(TypedDict):
    type: NotRequired[Literal["image"] | _ChatMsgMediaImageTypeAliases]
    image_url: _OpenAIFormatImageInnerT1 | _OpenAIFormatImageInnerT2
class _OpenAIFormatImageT2(TypedDict):
    type: NotRequired[Literal["image"] | _ChatMsgMediaImageTypeAliases]
    image: _OpenAIFormatImageInnerT1 | _OpenAIFormatImageInnerT2
_OpenAIFormatImage: TypeAlias = _OpenAIFormatImageT1 | _OpenAIFormatImageT2

class _OpenAIFormatAudioInner(TypedDict):
    data: str
    format: NotRequired[Literal["wav", "mp3"]]
class _OpenAIFormatAudioT1(TypedDict):
    input_audio: _OpenAIFormatAudioInner
    type: NotRequired[Literal["input_audio"] | _ChatMsgMediaAudioTypeAliases]
class _OpenAIFormatAudioT2(TypedDict):
    audio: _OpenAIFormatAudioInner
    type: NotRequired[Literal["input_audio"] | _ChatMsgMediaAudioTypeAliases]
_OpenAIFormatAudio: TypeAlias = _OpenAIFormatAudioT1 | _OpenAIFormatAudioT2

class _OpenAIFormatTextT1(TypedDict):
    type: NotRequired[Literal["text"]]
    text: str
class _OpenAIFormatTextT2(TypedDict):
    type: NotRequired[Literal["text"]]
    content: str
_OpenAIFormatText: TypeAlias = _OpenAIFormatTextT1 | _OpenAIFormatTextT2

_OpenAIFormatMedia: TypeAlias = _OpenAIFormatImage | _OpenAIFormatAudio
_OpenAIFormatMsgContent: TypeAlias = _OpenAIFormatMedia | _OpenAIFormatText

_ChatMsgMedia: TypeAlias = ChatMsgMedia | _OpenAIFormatMedia | Image | Audio | Video | PDF | str | Path

ChatMsgMedias: TypeAlias = dict[int, _ChatMsgMedia]
'''media contents included in this chat msg. Key is position index for media. {media_index: ChatMsgMedia, ...}'''
_ChatMsgMediasList: TypeAlias = Sequence[_ChatMsgMedia]
'''Alternative media contents included in this chat msg as a list, e.g. [{type: "image", "content": ...}, ...]'''

def detect_media_type(t: str)->ChatMsgMediaType|None:
    '''detect media type from type string, e.g. `image_url`. 
    Return None if not recognized.'''
    if not isinstance(t, str):
        return None
    t = t.lower().strip()
    image_file_types = ('jpg', 'png', 'bmp', 'tiff', 'webp')
    audio_file_types = ("wav", "mp3", "aac", "flac", "opus", "ogg", "m4a", "wma")
    video_file_types = ("mp4", "gif")
    
    if t in get_args(_ChatMsgMediaImageTypeAliases) or ('image' in t) or (t in image_file_types):
        return "image"
    elif t in get_args(_ChatMsgMediaAudioTypeAliases) or ('audio' in t) or (t in audio_file_types):
        return "audio"
    elif t in get_args(_ChatMsgMediaVideoTypeAliases) or ('video' in t) or (t in video_file_types):
        return "video"
    return None

class ChatMsg(TypedDict, total=False):
    """Single chat msg with no role."""

    content: Required[str]
    """
    Raw content of the chat msg.
    You can include media label within text content to identify the location,
    e.g. `<__MEDIA_0__>` to refer to the media with id=0.
    
    Also, it is allowed for you to pass a dict like `{'content': 'data:image/png;base64,...', 'type': 'image'}`.
    In that case, it will be turned into:
    ```
    {
        'content': '<__MEDIA_0__>',
        'medias': {
            0: {'type': 'image', 'content': 'data:image/png;base64,...'}
        }
    }
    
    You can also use <__media__>/<__MEDIA__> without index. In that case, the order of media contents 
    in `medias` will be used to identify the location.
    ```
    WARNING: Each media can at most have 1 tag in the content. Duplicate tags will be escaped.
    """
    textified_content: str|None
    """
    The textified content, i.e. image/audio is translated into meaningful text for
    inputting to non-multimodal LLM.
    
    This value will only be set when `get_text_content` is called. 
    WARNING: You should not pass this value directly except you know what you are doing.
    """
    timestamp: int
    """
    Timestamp this the chat msg. Default to be current time.
    This field will only be used in some special cases, e.g. packing as a conversation prompt.
    By default, it will not be used in normal LLM completion. 
    NOTE: the value should in ms, i.e. 13 digits.
    """
    multi_modal: bool|None
    """
    Whether passing media contents directly to multi-modal model(if available).
    This field will override `CompletionInput.multi_modal` field.
    """
    medias: _ChatMsgMedia | ChatMsgMedias | _ChatMsgMediasList
    '''media contents included in this chat msg. Key is media index.'''

class _TidiedChatMsg(ChatMsg):
    content: Required[str]
    textified_content: NotRequired[str|None]
    timestamp: NotRequired[int]
    multi_modal: NotRequired[bool|None]
    medias: dict[int, _TidiedChatMsgMedia]
    '''media contents included in this chat msg. Key is media index.'''

MsgMediaFormatter = TypeAliasType("MsgMediaFormatter", Callable[[Audio|Image|Video|PDF, "str"], str])
MsgMediaMeaningExtractor = TypeAliasType('MsgMediaMeaningExtractor', Callable[[Audio|Image|Video|PDF], str] | Callable[[Audio|Image|Video|PDF], Awaitable[str]])

def default_media_meaning_formatter(data: Image|Audio|Video|PDF, content: str) -> str:
    """the default implementation for formatting the extracted information from media type."""
    if not content:
        return ""
    if isinstance(data, Image):
        img = data if isinstance(data, Image) else None
        def html_img_tag(text: str):
            width = getattr(img, "width", None) if img else None
            height = getattr(img, "height", None) if img else None
            if not width or not height:
                return f'<img alt="{html.escape(text)}"/>'
            return f'<img alt="{html.escape(text)}" width="{width}" height="{height}"/>'
        return html_img_tag(content)
    elif isinstance(data, Audio):
        duration = data.duration_seconds if data else None
        def html_audio_tag(text: str):
            attrs = " "
            if text:
                attrs += f'aria-label="{html.escape(text)}" '
            if duration:
                attrs += f'duration="{duration}" '
            tag = f"<audio{attrs}/>"
            return tag
        return html_audio_tag(content)
    elif isinstance(data, Video):
        duration = data.duration if data else None
        def html_video_tag(text: str):
            attrs = " "
            if text:
                attrs += f'aria-label="{html.escape(text)}" '
            if duration:
                attrs += f'duration="{duration}" '
            tag = f"<video{attrs}/>"
            return tag
        return html_video_tag(content)
    elif isinstance(data, PDF):
        def html_pdf_tag(text: str):
            return f'<embed type="application/pdf">\n{html.escape(text)}\n</embed>'
        return html_pdf_tag(content)
    else:
        raise NotImplementedError(f"Unsupported type for text content: {type(content)}")

class _MediaText(BaseModel):
    text: str

async def default_media_meaning_extractor(media: Audio | Image | Video | PDF, client: "ThinkThinkSyn|None"=None)->str:
    """
    Extract the meaningful content from the media type.
    It will be called automatically when `get_text_content` is called.
    """
    if not client:
        if not (client:=current_thinkthinksyn_client.get(None)):
            raise RuntimeError("No thinkthinksyn client found in current context. Cannot extract media meaning.")
    if isinstance(media, Audio):
        prompt = 'Extract meaningful text content from the given audio, i.e. if there is speech, transcribe it; if there is music, describe it briefly.'
    elif isinstance(media, Image):
        prompt = 'Extract meaningful text content from the given image, i.e. if is a photo, describe it briefly; if there is text/paper/..., transcribe it.'
    elif isinstance(media, Video):
        prompt = 'Extract meaningful text content from the given video, i.e. if there is speech, transcribe it; if there is action, describe it briefly.'
    elif isinstance(media, PDF):
        raise NotImplementedError("Default media meaning extractor for PDF is not implemented yet.")
    else:
        raise NotImplementedError(f"Unsupported type for media: {type(media)}")
    result = await client.json_complete(
        [prompt, media],
        return_type=_MediaText,
    )
    if not result:
        raise RuntimeError("Failed to extract media meaning.")
    return result.text

def get_msg_media_media_obj(media: ChatMsgMedia) -> Audio | Image | Video | PDF:
    """return the proper media object."""
    media_cache = media.get("__media_content__", None)
    if media_cache:
        return media_cache  
    media_type = media.get('type', '')
    media_type = detect_media_type(media_type) if media_type else media_type
    media_content = media.get('content', '')
    if isinstance(media_content, (Image, Audio, Video, PDF)):
        return media_content
    if not media_type and media_content and isinstance(media_content, str):
        # tru guess type from content
        if media_content.startswith('data:'):
            prefix = media_content[5:].split(';')[0].lower()
            if 'image/' in prefix:
                media_type = "image"
            elif 'audio/' in prefix:
                media_type = "audio"
            elif 'video/' in prefix:
                media_type = "video"
            elif 'application/pdf' in prefix:
                media_type = "pdf"
        else:
            media_content_last_part = media_content.lower()[-5:].strip()
            if media_content_last_part.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                media_type = "image"
            elif media_content_last_part.endswith(('.wav', '.mp3', '.aac', '.flac', '.opus', '.ogg', '.m4a', '.wma')):
                media_type = "audio"
            elif media_content_last_part.endswith(('.mp4', '.gif', '.mov', '.avi', '.mkv')):
                media_type = "video"
            elif media_content_last_part.endswith('.pdf'):
                media_type = "pdf"
    
    if not media_type:
        raise ValueError("Cannot detect media type automatically. Please specify `type` field in ChatMsgMedia.")
    if media.get("is_receipt", False):
        raise ValueError("Cannot get media from a receipt ChatMsgMedia.")
    if media_type == "audio":
        media_obj = Audio.Load(media_content)   
    elif media_type == "image":
        media_obj = Image.Load(media_content)   
    elif media_type == "video":
        media_obj = Video.Load(media_content)   
    elif media_type == "pdf":
        media_obj = PDF.Load(media_content) 
    else:
        raise ValueError(f"Unknown chat msg type: {media_type}")
    media["__media_content__"] = media_obj  
    return media_obj

def tidy_single_media(media: _ChatMsgMedia)->_TidiedChatMsgMedia:
    if isinstance(media, (Audio, Image, Video)):
        media_type = type(media).__name__.lower()
        return dict(type=media_type, content=media)     
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
                    return dict(type='image', content=data, **media)      
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
        return dict(type=media_type, content=media_cls.Load(media))
    raise TypeError(f"Invalid media type: {type(media)}")

def tidy_msg_medias(medias: _ChatMsgMedia|_ChatMsgMediasList|ChatMsgMedias)->dict[int, _TidiedChatMsgMedia]:
    if isinstance(medias, (list, tuple)):
        return {i: tidy_single_media(m) for i, m in enumerate(medias)}
    elif isinstance(medias, dict):
        first_key = next(iter(medias.keys()), None)
        if not isinstance(first_key, int):
            # single `ChatMsgMedia` dict
            return {0: _tidy_single_media(medias)}  
        return {k: _tidy_single_media(v) for k, v in medias.items()}    
    elif isinstance(medias, (Image, Audio, Video, str, Path)):
        return {0: tidy_single_media(medias)}
    else:
        raise TypeError(f"Invalid medias type: {type(medias)}")
    
async def get_chat_msg_media_textified_content(
    media: ChatMsgMedia,
    formatter: MsgMediaFormatter = default_media_meaning_formatter,
    extractor: MsgMediaMeaningExtractor = default_media_meaning_extractor,
    reformat: bool = False,
):
    curr_textified_content = media.get('textified_content', None)
    if not curr_textified_content or reformat:
        media_cache = media.get("__media_content__", None)
        if not media_cache and media.get("is_receipt", False):
            raise ValueError("Cannot get text content from a receipt ChatMsgMedia.")
        meaning = await async_run_any_func(extractor, get_msg_media_media_obj(media))
        media['textified_content'] = await async_run_any_func(formatter, get_msg_media_media_obj(media), meaning)
    return media.get('textified_content', "")

async def format_chat_msg_to_string(
    msg: ChatMsg, media_formatter: SyncOrAsyncFunc[[int, ChatMsgMedia], str|None] | dict[int, str|None] | None = None
) -> str:
    """
    Format this message into a single string.
    Args:
        - media_formatter: a function to format the media content, or a dict mapping.
                        If `None`, all media tags will be removed from the content.
    """
    medias = tidy_msg_medias(msg.get('medias', {}))
    content = msg.get('content', "")
    return await _format_media(medias, content, media_formatter)    

def format_chat_msg_to_relative_time_str(
    msg: ChatMsg,
    to: int | float | datetime | None = None,
    full: bool = False,
) -> str:
    """
    Return a meaningful string to represent the relative time between this msg and the given time,
    e.g. `1 secs ago`, `3 mins ago`...

    Args:
        - to: the time to compare with. If None, the current time will be used.
        - full: if True, it will return like `X days Y hour Z minute ago`, otherwise
                just `X days ago`.
    """
    def msg_relative_time(msg, to: int | float | datetime | None = None) -> relativedelta:
        if not to:
            to = datetime.now()
        elif isinstance(to, (int, float)):
            to = datetime.fromtimestamp(to)

        from_time = msg.get("timestamp", int(time.time() * 1000))
        if isinstance(from_time, int):
            from_time_str = str(from_time)
            if len(from_time_str) == 13:
                from_time = from_time / 1000.0
        return relativedelta(datetime.fromtimestamp(from_time), to)
    
    relative_time = msg_relative_time(msg, to)
    traverse_order = ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]
    ago, r = None, ""
    for unit in traverse_order:
        value = getattr(relative_time, unit)
        if value != 0:
            if ago is None:
                ago = True if value < 0 else False
            r += f"{abs(value)} {unit} "
            if not full:
                break
            else:
                relative_time -= relativedelta(**{unit: value})
                if relative_time == relativedelta():
                    break
    if not r:
        return "just now"
    if ago:
        ret = r.strip() + " ago"
    else:
        ret = r.strip() + " later"
    ret = ret.replace("seconds", "secs").replace("minutes", "mins").replace("hours", "hrs")  # for making it shorter
    return ret.strip()
    
async def get_chat_msg_textified_content(
    msg: ChatMsg,
    formatter: MsgMediaFormatter = default_media_meaning_formatter,
    extractor: MsgMediaMeaningExtractor = default_media_meaning_extractor,
    reformat: bool = False,
)->str:
    """
    Get a single string for representing this chat msg.
    If the content is multi-media type(audio, img, ...),  meaningful scripts
    will be extracted by calling other models.

    Args:
        - formatter: the formatter for making extracted text content as a prompt for LLM.
        - extractor: the extractor for extracting meaningful content from media type.
        - re_format: If True, the final formatted content will be overridden by the new one.
                    Otherwise the previous formatted content will be returned.
    """
    msg_textified_content = msg.get('textified_content', None)
    medias = msg.get('medias', {})
    medias = tidy_msg_medias(medias)
    msg['medias'] = medias  
    if not reformat and msg_textified_content:
        return msg_textified_content
    if not medias:
        textified = msg.get('content', "")  # do nothing
    else:
        tasks = []
        orders = []
        media_objs = []
        for k, c in medias.items():
            orders.append(k)
            media_objs.append(c.media())  
            tasks.append(
                get_chat_msg_media_textified_content(
                    c,  
                    formatter=formatter,
                    extractor=extractor,
                    reformat=reformat,
                )
            )
        results = await asyncio.gather(*tasks)
        formatted_results = [formatter(m, r) for m, r in zip(media_objs, results)]  
        textified = await format_chat_msg_to_string(msg, {orders[i]: formatted_results[i] for i in range(len(orders))})

    msg['textified_content'] = textified
    return textified

def is_chat_msg(obj: Any)->TypeIs[ChatMsg]:
    '''Check whether the given object is a ChatMsg.'''
    if not isinstance(obj, dict):
        return False
    if 'content' not in obj:
        return False
    if 'medias' in obj or 'textified_content' in obj or 'timestamp' in obj:
        return True
    return 'type' not in obj  # to avoid being ChatMsgMedia

class ChatMsgWithRole(ChatMsg):
    role: NotRequired[str]
    '''role of the chat msg, e.g. `user`, `assistant`, `system`, ...
    If not given, `user` will be used as default role.
    '''

class _TidiedChatMsgWithRole(_TidiedChatMsg):
    role: NotRequired[str]
    
def is_chat_msg_with_role(obj: Any)->TypeIs[ChatMsgWithRole]:
    '''Check whether the given object is a ChatMsgWithRole.'''
    if not is_chat_msg(obj):
        return False
    return 'role' in obj
    
class _OpenAIChatMsgWithRole(TypedDict):
    role: Literal["system", "user", "assistant"] | str
    content: str | _OpenAIFormatMsgContent | Sequence[_OpenAIFormatMsgContent]

class CompletionConfig(TypedDict, total=False):
    """
    Config for completion generation. Compatible with OpenAI format.
    As most of nodes are using llama.cpp as inference engine,
    for more config details, you can refer to https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md
    """

    max_tokens: int
    """
    How many tokens to generate at most(also called `max_new_token`).
    This value will be adjusted automatically to fit the model's max token limit
    during `LLM.validate_input` method.
    """
    temperature: float | None
    """
    (0.0, 2.0], higher->more random, lower->more deterministic.
    Default is 0.7, which is the same as OpenAI.
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation.
    """
    
    frequency_penalty: float|None
    """
    Repeat alpha frequency penalty.
    [-2.0, 2.0], higher->more repetition, lower->less repetition.
    For difference between `presence_penalty` and `frequency_penalty`,
    see: https://blog.csdn.net/jarodyv/article/details/129062982
    """
    presence_penalty: float|None
    '''
    Repeat alpha presence penalty. >0 means a higher penalty discourages the 
    model from repeating tokens
    For difference between `presence_penalty` and `frequency_penalty`,
    see: https://blog.csdn.net/jarodyv/article/details/129062982
    '''
    repetition_penalty: float | None
    """
    Control the repetition of token sequences in the generated text.
    [1.0, +inf), higher->less repetition, lower->more repetition
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation.
    """
    
    top_k: int | None
    """[1, +inf), number of top tokens to choose from.
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation."""
    top_p: float | None
    """[0.0, 1.0], cumulative probability of top tokens to choose from.
    See https://blog.csdn.net/stephen147/article/details/140635578 for explanation."""
    min_p: float | None
    """[0.0, 1.0], minimum cumulative probability of top tokens to choose from"""
    
    stop: str | list[str] | tuple[str, ...]
    """stop token(s) for generation. If None, it will use the default stop token of the model."""
    ignore_eos: bool|None
    '''ignore end of stream token and continue generating.
    WARN: this may result in infinite generation, so use with caution.'''
    logit_bias: Sequence[tuple[str, float|bool]]| dict[str, float|bool] | None
    """
    Modify the likelihood of a token appearing in the generated text completion. 
    For example, use "logit_bias": [['Hello',1.0]] to increase the likelihood of the token 'Hello', 
    or "logit_bias": [['Hello',-1.0]] to decrease its likelihood. 
    
    By setting the value to false, e.g `[['Hello', False]]` ensures that the token is never produced. 
    The tokens can also be represented as strings, e.g. [["Hello, World!",-0.5]] will reduce the
    likelihood of all the individual tokens that represent the string Hello, World!, 
    just like the presence_penalty does. 
    
    NOTE: the second value's range is [-100, 100] when it is float.
    """

ToolChoiceMode: TypeAlias = Literal["none", "auto", "required", "required_one"]
"""
Equivalent to `tool_choice` in OpenAI(`required_one` is our extra options):
Modes:
    - `none`: No tool will be chosen. In this case, message will be returned instead of tool calling. 
    - `auto`: Automatically choose 1 or more tools, or not choosing any.
    - `required`: One or more tools must be chosen.
    - `required_one`: must choose 1 tool.
"""
ToolCallMode: TypeAlias = Literal["define", "call", "inline"]
"""
Mode of tool call:
- `define`: Define the input params of the tool(s) should be called. This is the default mode when the tool
            is chosen, which is the same as OpenAI.
- `call`: (For internal tools only) After defining the input params, call the tool(s) with the defined params
          and get the result from `value` field. Note that this is only available when the tool is found 
          within backend server (by inheriting the `LLMTool` class).
- `inline`: (For internal tools only) After calling the tool, the calling result will be built as prompt and
            pass into LLM again to enforce its ability answering user's question. This is the default mode in 
            `chat` service.
"""

class ToolConfig(TypedDict, total=False):
    """configs for `tools` in CompletionInput"""

    choose_mode: ToolChoiceMode
    """
    Equivalent to `tool_choice` in OpenAI(`required_one` is our extra options):
    Modes:
        - `none`: No tool will be chosen. In this case, message will be returned instead of tool calling. 
        - `auto`: Automatically choose 1 or more tools, or not choosing any. If a tool should be called multiple times,
                  it will also be chosen with multiple params.
        - `required`: One or more tools must be chosen.
        - `required_one`: must choose 1 tool.
    NOTE: this field will be ignored if `tool_force_chosen` is set.
    """
    tool_force_chosen: str | Sequence[str] | None
    """
    Manually set the final chosen tool. It should be a list of tool names or a single tool name.
    Your manual choice should be in the list of `tools` in `CompletionInput`.
    Note: In OpenAI's format, this field is also put under `tool_choice`. 
          In this case, the value will be redirected to `tool_chosen`.
    """
    call_mode: ToolCallMode
    """
    Mode of tool call (default to be `define`):
    - `define`: Define the input params for calling the tool(s).
                This is the default mode, which is the same as OpenAI.
    - `call`: Call the tool(s) with params given by LLM finally, and get the result from 
              `value` field of `ToolCallResult`. Note that this is only available 
              when the tool is found within backend server (by inheriting the 
              `LLMTool` class).
              3rd-party-tool-upload system will be supported in the future.
    - `inline`: After calling the tool, the calling result will be built as prompt and pass 
                into LLM again to enforce its ability answering user's question.
                This is the default mode in `chat` service.   
    """

ToolParamType: TypeAlias = Literal['required', 'optional', 'hidden', 'return']
'''Type of the tool parameter. `optional` is only available when the parameter has a default value.'''

class ToolParam(JsonSchema, total=False):
    param_type: ToolParamType
    '''Type of the parameter. Note: `optional` is only available when the parameter has a default value.'''

class ToolInfo(TypedDict):
    '''
    Information about a LLM tool, for sending to model. Compatible with OpenAI's `tools` field.
    You should fill detailed information to get better result in calling. 
    '''
    name: str
    '''name of the tool. This is an important field for meaningful selection.'''
    description: NotRequired[str|None]
    '''description of the tool. This is an important field for meaningful selection.'''
    internal_key: NotRequired[str|None]
    '''For internal tools only, a unique string. See `llm_tools` module for more details.'''    
    params: dict[str, ToolParam]
    '''
    parameters of the tool. Each key is the name of the parameter, and the value is the detail of the parameter.
    Tool params should follows the specification of json schema. See: https://json-schema.org/understanding-json-schema
    '''
    return_type: NotRequired[ToolParam|None]
    '''Return type of the tool, for adding extra information to LLM.'''
    tool_creation_params: NotRequired[dict[str, Any]]
    '''
    Parameters for creating the tool instance, e.g. API key, model name, etc.
    This field is only available for internal tools, and when mode in `call` or `inline`.
    '''

_ToolType = TypeAliasType("_ToolType", str|ToolInfo|LLMTool)

class CompletionInput(AIInput, total=False):
    '''Input for LLM completion.'''
    
    context_id: str
    '''
    The context id for this input.
    If not given, a random context id will be generated.
    '''
    prompt: str
    '''
    The prompt to pass to the model. 
    Note:
        - If both `prompt` and `history`(`message`) are given, `prompt` will be act 
            as the final user input and `history` will be act as the chat history.
        - If `send_prompt_directly` is True, `prompt` will be send to model directly. No modification will be made.
    '''
    prefix: str
    '''
    The prefix before generation(or say the suffix after the built prompt). 
    It is helpful for special usage, e.g. adding `{` as prefix will make LLM more likely to output a perfect json.

    Note: when `tools` are given, this field will be ignored.
    '''
    system_prompt: str|None
    '''
    the system prompt to pass to the model. 
    You can still manually add system prompt in `history` by `system` role.
    '''
    with_model_system_prompt: bool
    '''
    When `system_prompt` is None, whether to use model's default system prompt.
    Default to be True.
    '''
    messages: Sequence[ChatMsgWithRole | _OpenAIChatMsgWithRole]
    '''
    The chat messages. It is used for inputting the full chat history, e.g.
    ```[{"role": "user", "content": "Hello!"}, ...]```
    
    `history` is an alias of `messages`. 
    
    NOTE: If your prompt has been included in this field, then no need to enter in field `prompt` anymore.
    '''
    send_prompt_directly: bool
    '''
    When send_prompt_directly=True, `prompt` field will not be modified(i.e. history & system_prompt will be ignore.), 
    and it will be send to model directly.
    '''
    completion_start_role: str
    '''
    When building prompt, it will be ended with a role's start tag, e.g. 
    ```
    {"role": "user", "content": "Hello!"} -> "user: Hello!<end_of_input_token>model: "
    ```
    You can change such starting role by this field. This will be helpful when you are
    doing some special completions, e.g. jail breaking by role inversion.
    This is field only available when `send_prompt_directly` is False.
    '''
    config: CompletionConfig
    '''The config for generating completion.'''
    openai_compatible: bool
    '''
    Whether to return a openai compatible output.
    DEPRECATED WARNING: this field hasn't being maintained for a long time. Not recommended to use anymore.
    USAGE WARNING: when this field is True, the returned output will not be `CompletionOutput` anymore.
    '''
    tools: _ToolType|Sequence[_ToolType]|None
    '''
    The tools to use in this input. Each tool should be a `LLMToolInfo` object,
    and each param in tool should be described as JSON schema.
    You can also pass tool key directly for internal tools.
    
    Note: tools will be ignored when:
        - if `send_prompt_directly` is True.
        - if `stream`=True
        - if `tool_config.mode`==`none`.
    '''
    tool_config: ToolConfig
    '''
    Config for tool calling. This is only available when `tools` is given.
    Tool setting fields in OpenAI's format(e.g. tool_choice, ...) when be put to this config automatically
    if you don't specify them in `tool_config` field.
    '''
    json_schema: JsonSchema|dict[str, Any]|None
    '''
    Json schema restricting the model to reply.
    This is only available when there is any json-schema available nodes, e.g. llama.cpp.
    Otherwise, this field will be ignored.
    
    NOTE: `json_schema` usually just limit the response in a certain structure,
        i.e. you still need to prompt the model to output in that json format.
    '''
    multi_modal: bool
    '''
    This field determines whether to allow building multimodal input/output prompts.
    E.g. when a multi-modal model is deployed under a multi-modal supporting framework,
    when `multi_modal` is True, image/audio messages will not be passed to Img2Text/S2T service
    anymore, but will be passed to the model directly.
    
    If you wanna ensure using multi-modal model, you still need to pass `model_filter`, `model` or `node_filter`
    to make sure you will select a multi-modal node.
    '''
    
def is_completion_input(obj: Any)->TypeIs[CompletionInput]:
    '''Check whether the given object is a CompletionInput.'''
    if not isinstance(obj, dict):
        return False
    anyof_fields = ['prompt', 'system_prompt', 'messages', 'history']   # `history` is alias of `messages`
    for f in anyof_fields:
        if f in obj:
            return True
    return False

class ToolCallResult(TypedDict):
    '''result of the tool call. If mode=`call`, `value` will be attached to the return.'''
    
    name: str
    '''name of the tool. Note that your chosen tool must be included in `tools` field of `CompletionInput`.'''
    params: dict[str, Any]
    '''parameters for calling the tool, determined by LLM. {param_name, value}'''
    internal_key: str|None
    '''internal id for the tool. This value only exists when the tool is registered within thinkthinksyn backend.'''
    
    # for `call`/`inline` mode
    value: Any
    '''
    Result of the tool call. This field is only available when mode=`call`/`inline`.
    If tool is not found or running process is failed, error msg will be attached to `error` field.
    '''
    error: str|None
    '''
    If the calling process is failed, the error message will be attached here.
    This is only for mode=`call`/`inline`.
    '''

class CompletionOutput(AIOutput[CompletionInput]):
    text: str
    '''
    The final output text of the LLM. Note that the text is concatenated with 
    your given `prefix` in `CompletionInput`.
    '''
    input_token_count: int
    '''The number of tokens in the input prompt.'''
    output_token_count: int
    '''The number of tokens in the output completion.'''
    tool_call_results: list[ToolCallResult]|None
    '''result of tool callings(only when tools are used).'''

# region stream output types
class CompletionMessageStreamOutput(TypedDict):
    '''A chunk of completion message stream output.'''
    event: Literal["message"]
    '''The event type.'''
    data: str
    '''The delta text generated in this chunk.'''

class CompletionToolCallStreamOutput(TypedDict):
    '''A tool call request in stream output.'''
    event: Literal["tool_calling"]
    '''The event type.'''
    data: list[ToolCallResult]
    '''The tool call request data.'''

CompletionStreamOutput = TypeAliasType(
    "CompletionStreamOutput", 
    CompletionMessageStreamOutput | CompletionToolCallStreamOutput
) 
# endregion

# region prompt
_media_tag_pattern = re.compile(r"\<__MEDIA_(\d+)__\>")

def build_media_tag(id: int)->str:
    """Build a in-content media tag with the given id,
    i.e. `<__MEDIA_{id}__>`."""
    return f"<__MEDIA_{id}__>"

def _tidy_content(medias: Iterable[int]|dict[int, Any], content: str, extend_tags: bool=True)->str:
    if not medias:
        return content
    for id in medias:
        m = re.search(f"\\<__MEDIA_{id}__\\>", content)
        if not m:
            m = re.search(f'\\<__media_{id}__\\>', content)
        if not m:
            if (m:=re.search(f'\\<__media__\\>', content, re.IGNORECASE)) is not None:
                content = content[: m.start()] + build_media_tag(id) + content[m.end() :]
            elif extend_tags:
                content += "\n" + build_media_tag(id)  # add the media tag if not found
    return content

def tidy_media_indices(
    content: str, medias: dict[int, ChatMsgMedia], offset: int = 0, extend_tags: bool = True
) -> tuple[str, dict[int, ChatMsgMedia]]:
    """
    Tidy the media indices in the content.
    Tags will be re-index to be continuous from `offset`, referring to their existing order.
    Return the updated content and the tidied medias(with index changed).

    Args:
        - medias: a dict of media contents, where the key is the index of the media.
        - content: the content of the chat msg, which may contain media tags 
                    like `<__MEDIA_{idx}__>`(or <__media__> <__MEDIA__>... is also acceptable).
        - offset: the offset to start the media indices from. Default to be 0.
        - extend_tags: whether to extend the media tags in the content if not found.
    """
    content = _tidy_content(medias, content, extend_tags=extend_tags)  # extend media tags if not found
    tidied_medias = {}
    tidied_content = ""
    last_end = 0

    for m in _media_tag_pattern.finditer(content):
        id = int(m.group(1))
        matched_str = m.group(0)
        tidied_content += content[last_end : m.start()]

        if id not in medias:  # escape if the media is not in the medias
            tidied_content += "<\\" + matched_str[1:-1] + "\\>"
        else:
            # replace the media tag with the new one
            tidied_content += build_media_tag(len(tidied_medias) + offset)
            tidied_medias[len(tidied_medias) + offset] = medias[id]
        last_end = m.end()

    tidied_content += content[last_end:]
    return tidied_content, tidied_medias

async def _format_media(
    medias: "dict[int, ChatMsgMedia]",
    content: str,
    media_formatter: SyncOrAsyncFunc[[int, "ChatMsgMedia"], str|None] | dict[int, str|None] | None = None,
) -> str:
    assert isinstance(
        media_formatter, (dict, type(None), Callable)
    ), f"Invalid media formatter: {media_formatter}. It should be a function or a dict."

    content = _tidy_content(medias, content)

    async def format_replace(id: int, media: "ChatMsgMedia"):
        tag = build_media_tag(id)
        if media_formatter is None:
            replace_str = ""  # remove tag
        elif isinstance(media_formatter, dict):
            replace_str = media_formatter.get(id, "")
        else:
            replace_str = await async_run_any_func(media_formatter, id, media)  
        return tag, replace_str or ''

    replace_tasks = await asyncio.gather(*[format_replace(id, media) for id, media in medias.items()])
    replacements = {tag: (replace_str or "") for tag, replace_str in replace_tasks}
    content = re.sub(
        "|".join(re.escape(tag) for tag in replacements.keys()),
        lambda m: replacements[m.group(0)] if m.group(0) in replacements else m.group(0),
        content,
    )
    return content

_AcceptedPromptMedias: TypeAlias = ChatMsgMedias | Sequence[ChatMsgMedia| Audio | Image | Video] | ChatMsgMedia | Audio | Image | Video
_BuildPromptInpT: TypeAlias = Sequence[str | ChatMsg | ChatMsgMedia] | CompletionInput

def _tidy_prompt_inp_medias(medias: _AcceptedPromptMedias)-> dict[int, ChatMsgMedia]:
    if isinstance(medias, dict):
        tidied_medias = {}
        for k, v in medias.items():
            if not is_chat_msg_media(v):
                if isinstance(v, (Audio, Image, Video, PDF)):
                    v = create_chat_msg_media_from_media(v)
                else:
                    raise ValueError(f"Invalid media content: {v}")
            tidied_medias[k] = v
        return tidied_medias
    elif isinstance(medias, (list, tuple)):
        tidied_medias = {}
        for i, m in enumerate(medias):
            if not is_chat_msg_media(m):
                if isinstance(m, (Audio, Image, Video, PDF)):
                    m = create_chat_msg_media_from_media(m)
                else:
                    raise ValueError(f"Invalid media content: {m}")
            tidied_medias[i] = m
        return tidied_medias
    else:
        if not is_chat_msg_media(medias):
            if isinstance(medias, (Audio, Image, Video, PDF)):
                medias = create_chat_msg_media_from_media(medias)
            else:
                raise ValueError(f"Invalid media content: {medias}")
        return {0: medias}

class _PromptMediaDict(dict):

    _prompt: "Prompt"

    def __new__(cls, origin: dict, prompt: "Prompt"):
        data = super().__new__(cls, origin)
        data._prompt = prompt
        return data

    def __init__(self, origin: dict, prompt: "Prompt"):
        super().__init__(origin)

    def __setitem__(self, key, value):
        origin = dict.get(self, key, None)
        super().__setitem__(key, value)
        if value != origin:
            self._prompt.__hash_cache__ = None  

    def __delitem__(self, key):
        super().__delitem__(key)
        self._prompt.__hash_cache__ = None

    def pop(self, *args, **kwargs):
        r = super().pop(*args, **kwargs)
        self._prompt.__hash_cache__ = None
        return r

    def popitem(self, *args, **kwargs):
        r = super().popitem(*args, **kwargs)
        self._prompt.__hash_cache__ = None
        return r

    def clear(self):
        super().clear()
        self._prompt.__hash_cache__ = None

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._prompt.__hash_cache__ = None

    def __copy__(self):
        return _PromptMediaDict(self.copy(), self._prompt)

    def __deepcopy__(self, memo=None):
        new_dict = {}
        for idx, m in self.items():
            if isinstance(m, BaseModel):
                m = m.model_copy()
            new_dict[idx] = m
        return _PromptMediaDict(new_dict, self._prompt)

class Prompt(str):
    """
    Built prompt fixed for certain LLM, i.e. special tokens are included in the prompt.
    Each prompt contains a list of media contents, which each is associated with a media tag
    `<__MEDIA_{idx}__>` in the content.
    In some special cases, prompt will act as a conversation prompt which has no special
    tokens, but still contains media contents.

    You can use `Prompt.Build` to build a prompt from the given input messages.
    """

    medias: dict[int, _TidiedChatMsgMedia]
    """
    media contents in the prompt. The key refers to the index of the media content,
    with associated with the media tag `<__MEDIA_{idx}__>` in the prompt content.
    
    NOTE: 
    1. when prompt is built, media is re-indexed according to the order it appears in the content.
    2. `<__media_{idx}__>` / `<__media__>` / `<__MEDIA__>` are also acceptable tags.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validator(data):
            if isinstance(data, str) and not isinstance(data, Prompt):
                if data.startswith("{") and data.endswith("}"):
                    data = json.loads(data)
                else:
                    return cls(data)

            if isinstance(data, dict):
                if not (prompt := data.get("prompt", None)):
                    raise ValueError("Invalid data: no 'prompt' field found")
                medias = data.get("medias", {})
                assert isinstance(medias, dict), f"Invalid media contents: {medias}"
                tidied_medias = {}
                for k, v in medias.items():
                    if isinstance(k, str):
                        if k.isdigit():
                            k = int(k)
                        else:
                            raise ValueError(f"Invalid media index: {k}")
                    if not is_chat_msg_media(v):    
                        if isinstance(v, str):
                            v_strip = v.strip()
                            if v_strip.startswith("{") and v_strip.endswith("}"):
                                v = json.loads(v)
                        v = tidy_single_media(v)
                    tidied_medias[k] = v
                obj = cls(prompt, medias=tidied_medias)
                return obj

            return data

        def serializer(data):
            if isinstance(data, Prompt):
                return {"prompt": str(data), "medias": {k: dump_chat_msg_media(m) for k, m in data.medias.items()}}     
            return str(data)

        class PromptSchemaModel(BaseModel):
            prompt: str
            medias: dict[int, ChatMsgMedia]

        return create_pydantic_core_schema(validator, serializer, schema_model=PromptSchemaModel)

    def __new__(
        cls, 
        prompt: str, 
        medias: _AcceptedPromptMedias | None = None
    ) -> Self:
        """
        Args:
            - prompt: the built prompt string to be used.
            - medias: the media contents in the prompt, with the key refers to the index of the media content.
                    Note that each media should associate with a media tag `<__MEDIA_{idx}__>` in the prompt content.
        """
        medias = _tidy_prompt_inp_medias(medias or {})  
        assert isinstance(prompt, str), f"Invalid prompt: {prompt}"
        assert isinstance(medias, dict), f"Invalid media contents: {medias}"
        prompt, medias = tidy_media_indices(prompt, medias, offset=0)   
        obj = super().__new__(cls, prompt)
        obj.medias = medias     
        return obj

    @classmethod
    def PackConvPrompt(
        cls,
        input: _BuildPromptInpT, 
        with_time:bool = True, 
        time_format:str|Literal['relative', 'relative-full']="relative",
        role_replacements: dict[str, str]|None = None,
        msg_content_formatter: SyncOrAsyncFunc[[bool, ChatMsg], str]|None = None,
    )->Self:
        '''
        Pack whole chatting conversation into a single prompt.
        
        Args:
            input (str/ChatMsg/ChatMsgMedia/CompletionInput/Sequence[str/ChatMsg/ChatMsgMedia/ChatMsgWithRole]): 
                        The input object which contains all chatting history.
            with_time (bool): Whether to include the time information in each msg.
            time_format: The format of time string, e.g. "%Y-%m-%d %H:%M:%S".
                        This field is only used when `with_time` is True.
                        If `relative`/`relative-full` is used, the time will be shown as relative time,
                        e.g. `2 days ago` or `2 days 3 hours ago`.
            role_replacements: A dict to replace role names in the prompt, e.g. {"assistant": "you"}
                                Note that the matching will be case-insensitive.
            msg_content_formatter: A function to format the message content. If None, all tags will be
                                removed from the message content.
        
        Why should I use this?
            LLM can only chat with user/system role, but it is hard to prompt special information
            to model in normal chatting format, e.g. RAG, memories. In some cases, role can even
            be more than 1, e.g. user_a, user_b, ...
            
            To complete such conversation, we have to pack the whole conversation as a single prompt
            to put within `system` role. This is help preventing jail-breaking.
            
            This is also the way we do conversation in `Chatter` class.
        '''
        if is_completion_input(input):
            input = input.copy()
            input.pop('prefix', None)  # ignore prefix in `PackConvPrompt`
        role_formatter = lambda s, _: str(s).strip()
        if not msg_content_formatter:
            async def default_msg_content_formatter(mtmd: bool, msg: ChatMsg)->str:
                if not mtmd:
                    return await get_chat_msg_textified_content(msg)
                else:
                    # do nothing, keep media tags, since we dont know what model will be used finally.
                    return msg['content']
            msg_content_formatter = default_msg_content_formatter
        
        return run_any_func(
            Prompt.Build,
            input=input,
            with_special_tokens=False, 
            with_model_system_prompt=False,
            last_beginning_role=None,
            with_time=with_time,
            time_format=time_format,
            role_suffix=': ',
            role_formatter=role_formatter,
            each_msg_suffix='\n',
            role_replacements=role_replacements,
            msg_content_formatter=msg_content_formatter,
        )   
        
    @classmethod
    async def Build(
        cls,
        input: _BuildPromptInpT,
        with_special_tokens: bool = True,
        with_model_system_prompt: bool = True,
        last_beginning_role: str | None = LLM_DEFAULT_ASSIST_ROLE_NAME,
        with_time: bool = False,
        time_format: str | Literal["relative", "relative-full"] = "%Y-%m-%d %H:%M:%S",
        role_replacements: dict[str, str] | None = None,
        role_prefix: str = "",
        role_suffix: str = "",
        role_formatter: Callable[[str, type["LLM"]], str] | None = None,
        inner_msg_prefix: str = "",
        inner_msg_suffix: str = "",
        each_msg_prefix: str = "",
        each_msg_suffix: str = "",
        model: "str|LLM|type[LLM]|None" = None,
        msg_content_formatter: SyncOrAsyncFunc[[bool, ChatMsg], str] | None = None,
    ) -> Self:
        """
        Build a prompt from the given input messages.

        Args:
            - input: the input messages to build the prompt from. It can be:
                    - `str`: a single prompt string. It will be seen as a user message.
                    - `ChatMsgMedia`: a single media message, which will be treated as a user message(but no content, just media).
                    - `CompletionInput`: an CompletionInput object, which contains messages and other parameters.
                                 if `CompletionInput.send_prompt_directly` is True, no building will be
                                done, and this method will return the prompt directly.
                    - `Sequence[ChatMsgWithRole|ChatRecord|str|ChatMsgMedia]`: a sequence of messages
            - with_special_tokens: whether to include model's special tokens in the prompt.
                                  This is only available when `model` is given.
            - with_model_system_prompt: when model is given, and the first message is not a system message,
                                      You can set this to True to include the model's default system prompt
                                      into the built prompt.
            - last_beginning_role: whether to add the a role beginning at the end of the prompt,
                                    e.g. `assistant:`. This is for building completion prompt. If none, no role will be added.
            - with_time: whether to include the time into each message, e.g. `user(2023-10-01 12:00:00): Hello`.
            - time_format: the format of time string. Default is "%Y-%m-%d %H:%M:%S". This is only used when `with_time` is True.
                           If `relative`/`relative-full` is used, the time will be shown as relative time, e.g. `user (2 days ago): Hello`.
                          See `ChatMsgWithRole.relative_time_str` for more details.
            - role_replacements: a dict to replace role names in the prompt, e.g. {"assistant": "you"}
                            Note that the matching will be case-insensitive.
            - role_prefix: the prefix before the role name in the prompt
            - role_suffix: the suffix after the role name in the prompt
            - role_formatter: a function to format the role name in the prompt.
            - inner_msg_prefix: the prefix before the message content in the prompt(but within start content token)
            - inner_msg_suffix: the suffix after the message content in the prompt(but within end chat token)
            - each_msg_prefix: the prefix before each message in the prompt
            - each_msg_suffix: the suffix after each message in the prompt, e.g. "\n"
            - model: the model to build the prompt for. It is used for formatting roles, escaping special tokens,
                    and adding special tokens to the prompt. If None, the default LLM will be used.
                    If a string is given, it will be treated as the model name to search for.
            - msg_content_formatter: ```(mtmd: bool, msg: ChatMsg) -> str.``` A function to format the message content.
                                    You can decide how to format media labels within the content. If None, all
                                    labels will be removed from the content.
        """
        if isinstance(input, Prompt):
            raise ValueError("Input is already a Prompt object, no need to build again.")

        if isinstance(model, str):
            real_model: _CommonLLM|type[_CommonLLM]|None = LLM.SearchCls(model)
        elif isinstance(model, type) and issubclass(model, LLM):
            if model == LLM:
                real_model = None
            else:
                real_model = model  
        elif isinstance(model, LLM):
            if type(model) == LLM:
                real_model = None
            else:
                real_model = model  
        else:
            real_model = None
        
        if with_model_system_prompt:
            model_system_prompt = getattr(model, 'DefaultSystemPrompt', None)
        else:
            model_system_prompt = None
        messages: list[ChatMsgWithRole] = []
        role_matched_cache = {}
        mtmd = True
        medias = {}
        if with_special_tokens:
            prompt = (real_model.PromptStartToken.format % real_model.PromptStartToken.token) if (real_model and real_model.PromptStartToken) else ""
        else:
            prompt = ""

        # region helper functions
        def _match_role(r: str) -> str | None:
            if not role_replacements:
                return None
            if r in role_matched_cache:
                return role_matched_cache[r]
            build_pattern = lambda r: r"^[<>\?\n\!:\|\s]{0,4}" + r.lower() + r"[<>\?\n\!:\|\s]{0,4}$"
            r = full_width_text_tidy_up(r.lower()).strip()
            for role_matcher, replace in role_replacements.items():
                pattern = build_pattern(role_matcher)
                if re.search(pattern, r, re.IGNORECASE):
                    role_matched_cache[r] = replace
                    return replace
            return None

        async def mtmd_format_msg(id: int, msg_media: ChatMsgMedia):
            tag = build_media_tag(id)
            msg_type = msg_media.get('type', None)
            if msg_type:
                msg_type = detect_media_type(msg_type)
            if msg_type == "image":
                if real_model and not real_model.SupportsImage:
                    return await get_chat_msg_media_textified_content(msg_media)
            elif msg_type == "audio":
                if real_model and not real_model.SupportsAudio:
                    return await get_chat_msg_media_textified_content(msg_media)
            return tag

        async def _default_msg_formatter(mtmd: bool, msg: ChatMsgWithRole) -> str:
            if not mtmd:
                content = await get_chat_msg_textified_content(msg)
            else:
                content = await format_chat_msg_to_string(msg, mtmd_format_msg)
            return content

        _msg_formatter = msg_content_formatter or _default_msg_formatter

        async def format_msg(idx: int, msg: ChatMsgWithRole) -> str:
            msg_medias = msg.get('medias', {})
            msg_content = msg.get('content', '')
            multi_modal = msg.get('multi_modal', None)
            role_str = msg.get('role', LLM_DEFAULT_USER_ROLE_NAME)
            if not msg_medias:
                content = msg_content
            else:
                _mtmd = multi_modal if multi_modal is not None else mtmd
                content = await async_run_any_func(_msg_formatter, _mtmd, msg)  
            
            if matched_role := _match_role(role_str):
                role_str = matched_role
            if with_time:
                if time_format in ("relative", "relative-full"):
                    time_str = format_chat_msg_to_relative_time_str(msg, full=(time_format == "relative-full"))
                else:
                    msg_timestamp = msg.get('timestamp', int(time.time() * 1000))
                    time_str = datetime.fromtimestamp(msg_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                _role_suffix = f"({time_str}){role_suffix}"
            else:
                _role_suffix = role_suffix

            with_end_token = with_special_tokens
            if idx == len(messages) - 1 and last_beginning_role:
                with_end_token = False
            formatted = model.FormatMsg(
                role=role_str,
                msg=content,
                role_prefix=role_prefix,
                role_suffix=_role_suffix,
                role_formatter=role_formatter,
                include_start_token=with_special_tokens,
                include_content_start_token=with_special_tokens,
                include_end_token=with_end_token,
                msg_prefix=inner_msg_prefix,
                msg_suffix=inner_msg_suffix,
            )
            _each_msg_suffix = each_msg_suffix
            if last_beginning_role and idx == len(messages) - 1:
                _each_msg_suffix = ""
            return each_msg_prefix + formatted + _each_msg_suffix

        # endregion

        if is_completion_input(input):
            inp_messages = input.get('messages', input.get('history', None))
            inp_prompt = input.get('prompt', '').strip()
            if input.get('send_prompt_directly', False):
                if not inp_prompt:
                    if inp_messages:
                        _logger.warning(f"CompletionInput.send_prompt_directly is True, but no prompt provided. Using messages instead.")
                        input['send_prompt_directly'] = False
                    else:
                        raise ValueError("CompletionInput.send_prompt_directly is True, but no prompt or messages provided.")
                return cls(inp_prompt)

            mtmd = input.get('multi_modal', False)
            system_prompt = input.get("system_prompt", None)
            if isinstance(system_prompt, str):
                system_prompt = system_prompt.strip()

            if inp_messages:
                if system_prompt and not (
                    system_prompt == inp_messages[0].content.strip() and inp_messages[0].is_system_role(model)
                ):
                    messages.insert(0, ChatMsgWithRole(role=model.DefaultSystemRole.name, content=system_prompt))
                elif (
                    model_system_prompt
                    and not input.messages[0].is_system_role(model)
                    and (input.messages[0].content.strip() != model_system_prompt.strip())
                ):
                    messages.insert(
                        0, ChatMsgWithRole(role=model.DefaultSystemRole.name, content=model_system_prompt.strip())
                    )
            else:
                if (sys_prompt := (system_prompt or model_system_prompt)) and sys_prompt.strip():
                    messages.append(ChatMsgWithRole(role=model.DefaultSystemRole.name, content=sys_prompt.strip()))

            if input.messages:
                messages.extend([m.model_copy() for m in input.messages])
            if input.prompt.strip() and not (
                len(messages) > 0
                and (input.prompt.strip() == messages[-1].content.strip())
                and (messages[-1].role in model.DefaultUserRole.alias)
            ):
                messages.append(ChatMsgWithRole(role=model.DefaultUserRole.name, content=input.prompt))
        else:
            if check_value_is(input, (str, ChatMsgMedia, ChatMsg)):
                input = [input]  
            if isinstance(input, Sequence):
                for i, msg in enumerate(input):
                    if check_value_is(msg, str):
                        msg = ChatMsgWithRole(role=model.DefaultUserRole.name, content=msg)
                    if check_value_is(msg, ChatMsgMedia):
                        msg = ChatMsgWithRole(role=model.DefaultUserRole.name, medias={0: msg})
                    if check_value_is(msg, ChatMsg) and not check_value_is(msg, ChatMsgWithRole):
                        msg = msg.to_role_msg(model.DefaultUserRole.name)
                    if check_value_is(msg, ChatMsgWithRole):
                        if (
                            i == 0
                            and model_system_prompt
                            and not msg.is_system_role(model)
                            and msg.content.strip() != model_system_prompt.strip()
                        ):
                            messages.insert(
                                0,
                                ChatMsgWithRole(role=model.DefaultSystemRole.name, content=model_system_prompt.strip()),
                            )
                        messages.append(msg.model_copy())
                    else:
                        raise ValueError(
                            f"Invalid input type: {type(msg)}. Expected str, ChatMsgMedia, ChatMsg or ChatMsgWithRole."
                        )
            else:
                raise ValueError(
                    f"Invalid input type: {type(input)}. Expected str, ChatMsgMedia, ChatMsg or ChatMsgWithRole."
                )
        if last_beginning_role:
            messages.append(ChatMsgWithRole(role=last_beginning_role, content=""))

        msg_format_tasks = []
        for i, single_msg in enumerate(messages):
            single_msg.tidy_media_indices(offset=len(medias))
            medias.update(single_msg.medias)
            msg_format_tasks.append(format_msg(i, single_msg))

        msgs = await asyncio.gather(*msg_format_tasks)
        prompt += "".join(msgs)

        for id in tuple(medias.keys()):
            if not re.search(re.escape(build_media_tag(id)), prompt):
                # means that the tag already replaced by meaningful text
                medias.pop(id, None)

        prompt, medias = tidy_media_indices(prompt, medias, offset=0)

        if last_beginning_role and isinstance(input, CompletionInput) and input.prefix:
            if isinstance(input.prefix, str):
                prompt += input.prefix
            elif isinstance(input.prefix, CompletionPrefix):
                prompt += input.prefix.inject(input)
            else:
                raise ValueError(f"Invalid prefix type: {type(input.prefix)}")

        return cls(prompt, medias=medias)

    async def get_textified_content(self):
        """
        Alias of `format_media` method, which formats the prompt into meaningful text content
        by img2text, audio2text, etc.
        """
        async def formatter(idx: int, media: ChatMsgMedia) -> str|None:
            return await get_chat_msg_media_textified_content(media)
        return await self.format_media(media_formatter=formatter)

    async def format_media(
        self, media_formatter: SyncOrAsyncFunc[[int, ChatMsgMedia], str|None] | dict[int, str|None] | None = None
    ) -> str:
        """
        format the prompt into a single string, and replace
        media tags with the formatted media content.

        Args:
            - media_formatter: a function to format the media content, or a dict mapping.
                            If `None`, all media tags will be removed from the content.
        """
        return await _format_media(self.medias, str(self), media_formatter)

    def re_order_medias(self, offset: int = 0) -> Self:
        """
        Re-order the media indices in the prompt with the given offset,
        e.g.
        ```
        {321: {...}, 456: {...}} -> {0: {...}, 1: {...}} (when offset=0)
        ```
        Tags within the content will also be changed accordingly.
        NOTE: this returning prompt is a new object, not the original one.
        """
        tidied_content, tidied_medias = tidy_media_indices(str(self), self.medias, offset)
        return self.__class__(tidied_content, medias=tidied_medias)  

    def to_role_msg(self, role: str = LLM_DEFAULT_USER_ROLE_NAME) -> "ChatMsgWithRole":
        """Convert this prompt to a `ChatMsgWithRole` with the given role."""
        return ChatMsgWithRole(
            role=role,
            content=str(self),
            medias=self.medias,     
        )

    def to_completion_input(
        self,
        role: str = LLM_DEFAULT_SYSTEM_ROLE_NAME,
        **kwargs,
    ) -> "CompletionInput":
        """
        Convert this prompt to an `ChatMsgWithRole` with the given role,
        and then put into a new `CompletionInput` object.
        `**kwargs** will be passed to the `CompletionInput` constructor. You
        can use this to set other parameters.
        """
        msg = self.to_role_msg(role)
        msgs_field_aliases = ("messages", "history")
        for a in msgs_field_aliases:
            if a in kwargs:
                msgs = kwargs.pop(a)
                break
        else:
            msgs = []
        assert isinstance(msgs, (list, tuple)), f"Invalid messages: {msgs}. Expected list or tuple."
        msgs = list(msgs)
        if role.lower() in LLM_DEFAULT_SYSTEM_ROLE_ALIAS:
            msgs.insert(0, msg)  # insert the prompt as the first message
        else:
            msgs.append(msg)
        kwargs["messages"] = msgs
        return CompletionInput(**kwargs)

    def __setattr__(self, name: str, value):
        if name == "medias":
            if isinstance(value, dict) and not isinstance(value, _PromptMediaDict):
                value = _PromptMediaDict(value, self)
            self.__hash_cache__ = None  # reset the hash cache when medias are set
        return super().__setattr__(name, value)

    def __hash__(self) -> int:
        if not getattr(self, "__hash_cache__", None):
            def formatter(_, media: ChatMsgMedia) -> str:
                content = media.get('content', '')
                if len(content) > 1024:
                    content = str(hash(media.content))
                else:
                    content = media.content
                return f"{media.type}+{content}"
            formatted = run_any_func(self.format_media, media_formatter=formatter)
            self.__hash_cache__ = hash(formatted)
        return self.__hash_cache__  

    def __str__(self) -> str:
        return super().__str__()  

    def __repr__(self) -> str:
        prompt_for_log = str(self)
        if len(prompt_for_log) > 128:
            prompt_for_log = prompt_for_log[:64] + " ... " + prompt_for_log[-64:]
        medias_for_log = {}
        for k, v in self.medias.items():
            if len(v.get("content", '')) > 128: 
                content = v['content'][:64] + " ... " + v['content'][-64:]  
            else:
                content = v.get("content", '')   
            medias_for_log[k] = self.medias[k].model_copy(update={'content': content})
        return f"Prompt(prompt={prompt_for_log}, medias={medias_for_log})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Prompt):
            return str(self) == str(other) and self.medias == other.medias
        elif isinstance(other, str):
            return str(self) == other
        return False

    def __add__(self, other: str | Self) -> Self:
        if isinstance(other, Prompt):
            op, opm = str(other), other.medias
            op, opm = tidy_media_indices(op, opm, offset=len(self.medias))
            return self.__class__(str(self) + op, medias={**self.medias, **opm})  
        elif isinstance(other, str):
            return self.__class__(str(self) + other, medias=self.medias)  
        raise TypeError(f"Cannot add {type(other)} to Prompt")
# endregion


__all__ = [
    'ChatMsgMediaType',
    'ChatMsgMedia',
    'ChatMsgMedias',
    'detect_media_type',
    'ChatMsg',
    'ChatMsgWithRole',
    'CompletionConfig',
    'is_chat_msg_media',
    'is_chat_msg',
    'is_chat_msg_with_role',
    
    'ToolChoiceMode',
    'ToolParamType',
    'ToolParam',
    'ToolCallMode',
    'ToolConfig',
    'ToolInfo',
    'ToolCallResult',
    
    'CompletionInput',
    'CompletionOutput',
    'is_completion_input',

    'CompletionStreamOutput',
    'CompletionMessageStreamOutput',
    'CompletionToolCallStreamOutput',
    
    'build_media_tag',
    'tidy_media_indices',
    'Prompt',
]