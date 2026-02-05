from typing import Any, TypedDict

from .base import AIInput, AIOutput
from .models import StreamableAudioFormat
from ..common_utils.data_structs import Audio, AudioFormat

class T2SInput(AIInput, total=False):
    text: str
    """
    Text to be converted into speech. 
    String must not be empty after strip.
    """
    language: str|None
    """
    Target language for the speech. If not specified, it will be determined by the text.
    Language text passed in will be forced to lower so as to be case-insensitive.
    
    You should confirm the language is supported by the model before inference,
    otherwise the default language of the model will be used.
    """
    prompt: str|None
    '''prompt for T2S model (if supported, e.g. CosyVoice).'''
    audio_prompt: Audio|None
    '''
    Audio prompt for T2S model (if supported, e.g. CosyVoice).
    This audio gives instructions about the desired style, speaker, and acoustic environment.
    '''
    speed: float
    """The speed of the speech. 1.0 means normal speed."""
    speaker: str | None
    """
    The speaker of the speech. If not specified, the default speaker of the model will be used.
    You must ensure the given speaker name is included in the model's available speakers, 
    otherwise the default speaker will be used.
    NOTE: some model may not support this field.
    """
    format: AudioFormat|StreamableAudioFormat
    """
    The format of the response audio. Equivalent to OpenAI's `response_format` field.
    Note: stream mode's format is limited.
    """
    return_bytes: bool
    """
    Whether to return the audio content in response directly(acts like OpenAI's return).
    Default is `False`, and data will included in the `data` field(base64 encoded) of the response dict.
    Note: 
        * this field can be seen as equivalent to the logic of `openai_compatible`
        * this field only acts on real deployment(i.e. service system is enabled). It will not affect the local test.
    """
    auto_detect_lang: bool
    """Whether to auto detect the language of the text. This field only works when `language` is not specified"""
    
class T2SStreamOutput(TypedDict):
    data: str
    """The (wav) chunk data of the speech, encoded in base64."""
    channels: int
    """The number of channels of the audio"""
    sampling_rate: int
    """The sampling rate of the audio"""
    sampling_width: int
    """The sample width of the audio"""
    format: StreamableAudioFormat
    
    model: str|dict[str, Any]|None
    """
    The final model used for generating this t2s output.
    NOTE: though `model` is annotated as `T2SModel|None`, but logically it will 
          NEVER be None.
    """
    token_count: int
    """The output token count of this chunk. Tokens has converted to wav bytes."""
    
class T2SOutput(AIOutput):
    data: str
    """
    Base64 encoded audio data, with header included, and converted to the target format
    set in `T2SInput.format`.
    """
    token_count: int
    """
    The number of output tokens, for counting usage.
    This may not be supported by some models.
    """
    
__all__ = [
    'T2SInput',
    'T2SOutput',
    'T2SStreamOutput',
]