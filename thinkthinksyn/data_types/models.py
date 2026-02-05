if __name__ == "__main__":  # for debugging
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'thinkthinksyn.data_types'

from functools import cache
from dataclasses import dataclass
from typing import (Never, TypeAlias, Literal, Type, Self, TypeVar, Generic, ClassVar, Callable, Union, 
                    Sequence, Final)

from .base import ConditionProxy
from ..common_utils.decorators import class_property
from ..common_utils.data_structs import CaseIgnoreStrSet

# region bases
_ServiceToModelMatcher: dict[str, list[Type['_AIModel']]] = {}
'''{service_type: [model_classes]}'''

class _AIModel:
    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a data type class and cannot be instantiated.")
    
    def __init_subclass__(cls, type: str|None=None) -> None:
        type = type or getattr(cls, 'ServiceType', None)
        cls.ServiceType = type
        if type:
            if type not in _ServiceToModelMatcher:
                _ServiceToModelMatcher[type] = []
            if cls not in _ServiceToModelMatcher[type]:
                _ServiceToModelMatcher[type].append(cls)
    
    ServiceType: ClassVar[str|None] = None
    '''service type of the model, e.g. 'completion', 't2s', 's2t', `img2txt`, 'embedding', ...'''
    
    Name = ConditionProxy[str, Never]('Name')
    '''model name registered in thinkthinksyn backend'''
    Alias = ConditionProxy[tuple[str, ...], str]('Alias')
    '''model alias, i.e. other names for the model'''
    
    @class_property
    def ClassName(cls)->str:
        '''get the class name of the model class.'''
        return cls.__name__     # type: ignore
    
    @classmethod
    @cache
    def SearchCls(cls, name_or_alias: str)->Type[Self]|None:
        '''
        Search for the model class by its Name, class name, or Alias. Return None if not found.
        If this method is called with a bound model class with specific service type,
        only models of the same service type will be searched.
        
        NOTE: fuzzy matching will be performed, i.e. not case sensitive and ignore underscores/dashes.
        '''
        def fuzzy_sim(s):
            if isinstance(s, str):
                return s.replace('_', '').replace('-', '').lower()
            return s
        name_or_alias = fuzzy_sim(name_or_alias)
        if fuzzy_sim(cls.Name) == name_or_alias or fuzzy_sim(cls.ClassName) == name_or_alias or \
              name_or_alias in map(fuzzy_sim, getattr(cls, 'Alias', tuple())):
            return cls
        if (service_type := getattr(cls, 'ServiceType', None)):
            candidates = _ServiceToModelMatcher.get(service_type, [])
        else:
            candidates = cls.__subclasses__()
        for model_cls in candidates:
            if fuzzy_sim(model_cls.Name) == name_or_alias or fuzzy_sim(model_cls.ClassName) == name_or_alias or \
                  name_or_alias in map(fuzzy_sim, getattr(model_cls, 'Alias', tuple())):
                return model_cls    # type: ignore
        return None

class _ChildAIModel(_AIModel):
    Name: str
    Alias: tuple[str, ...] = tuple()
    
_CT = TypeVar("_CT", bound='_AIModel')

class _DefaultableAIModel(Generic[_CT], _AIModel):
    '''
    Special base class for models that has a default option for unspecified inputs.
    Default model must specify a fake [_T] to for better type hinting.
    '''

    @classmethod
    def Default(cls)->Type[_CT]:
        '''get the default model class for this model type.'''
        if not (m := getattr(cls, '__DefaultModel__', None)):
            raise NotImplementedError(f"{cls.__name__} has no default model defined.")
        return m
    
    @classmethod
    def RegisterDefault(cls, model_cls: Type[Self])->None:
        '''
        Register the default model class for this model type.
        In client side, you can also override the default model, thus you can call AI services
        without specifying model or model_filter.
        '''
        if not issubclass(model_cls, cls):
            raise TypeError(f"model_cls must be a subclass of {cls.__name__}.")
        setattr(cls, '__DefaultModel__', model_cls)

class LLM(_AIModel, type='completion'): 
    '''
    AI large language model(LLM) data type with condition proxies for its attributes.
    You can use these condition proxies to filter LLM models in thinkthinksyn backend,
    e.g.
    ```python
    async def test():
        return (await tts.completion(
            prompt='1+1? tell me ans directly without other words.',
            model_filter=((LLM.Name == Gemma3_27B_Instruct.Name) | ('qwen' in LLM.Alias))
            ...
        ))['text']
    ```
    '''
    
    B = ConditionProxy[int, Never]('B')
    '''model size in billions of parameters'''
    MoE = ConditionProxy[bool, Never]('MoE')
    '''whether this is a MoE model(mixture of experts)'''
    SupportsImage = ConditionProxy[bool, Never]('SupportsImage')
    '''whether the model supports image input/output'''
    SupportsAudio = ConditionProxy[bool, Never]('SupportsAudio')
    '''whether the model supports audio input/output'''
    SupportsVideo = ConditionProxy[bool, Never]('SupportsVideo')
    '''whether the model supports video input/output'''
    
    DefaultUserRole = ConditionProxy[str, str]('DefaultUserRole')
    '''The default user role. It is used for the user role in chat.'''
    DefaultAssistRole = ConditionProxy[str, str]('DefaultAssistRole')
    '''The default assist role. It is used for the assistant role in chat.''' 
    DefaultSystemRole = ConditionProxy[str, str]('DefaultSystemRole')
    '''The default system role. It is used for the system role in chat.'''
    
    DefaultSystemPrompt = ConditionProxy[str|None, str]('DefaultSystemPrompt')
    """
    The default system prompt. It is used for the first system prompt in the chat.
    If this field is not set, system prompt will not be automatically added.
    
    Note: this field can be override by `system_prompt` in input.
    """
    PromptStartToken = ConditionProxy[str|None, str]('PromptStartToken')
    """
    The string for replacing the `prompt start token`. `Prompt start token` is the beginning token of the whole prompt,
    e.g. `<|begin_of_text|>` in llama3. 
    
    You can set `format` field in SpecialToken to add line breaks or spaces for formatting the correct prompt.
    """
    ChatStartToken = ConditionProxy[str|None, str]('ChatStartToken')
    """
    The string for replacing the `chat start token`. `Chat start token` is the token to start chat,
    e.g. `<|im_start|>` in chatml format, <|start_header_id|> in llama3.
    
    You can set `format` field in SpecialToken to add line breaks or spaces for formatting the correct prompt.
    """
    ChatContentStartToken = ConditionProxy[str|None, str]('ChatContentStartToken')
    """The string for replacing the `chat content start token`. 
    `Chat content start token` is the token just before the msg content (after the role string), 
    e.g. `<|end_header_id|>` for llama3. 
    
    You can set `format` field in SpecialToken to add line breaks or spaces for formatting the correct prompt.
    """
    ChatEndToken = ConditionProxy[str|None, str]('ChatEndToken')
    """
    The string for replacing the `chat end token`. `Chat end token` is token when a msg is ended,
    e.g. `<|im_end|>` in chatml format. This is also be used for detecting stop.
    
    Note: When this field is None, it will be decided automatically with the model setting on huggingface(in case the model name is correct).
          It will also be the default value for `StopTokens`.
    
    You can set `format` field in SpecialToken to add line breaks or spaces for formatting the correct prompt.
    """
    StopTokens = ConditionProxy[set[str], str]('StopTokens')
    """
    The stop tokens for the model. This field will be extend with `ChatEndToken` automatically.
    Note: This is a set of STRING, not `SpecialToken`, since it is mainly for making configs for generation.
    """

class _ChildEmbeddingModel(_ChildAIModel):
    DefaultEmbeddingDim: int

class EmbeddingModel(_DefaultableAIModel[_ChildEmbeddingModel], type='embedding'):
    '''
    Embedding model data type with condition proxies for its attributes.
    You can use these condition proxies to filter embedding models in thinkthinksyn backend,
    e.g.
    ```python
    async def test():
        return (await tts.embedding(
            text='hello world',
            model_filter=(EmbeddingModel.Name == ZPointLarge.Name)
            ...
        ))['embedding']
    ```
    '''
    DefaultEmbeddingDim = ConditionProxy[int, Never]('DefaultEmbeddingDim')
    '''default embedding dimension of the model'''

class RerankingModel(_AIModel, type='rerank'):
    MaxTokens = ConditionProxy['int|None', Never]('MaxTokens')
    '''maximum tokens that the model can process. None for no limitation.'''
    
class S2TModel(_AIModel, type='s2t'):
    '''
    Speech-to-Text model data type with condition proxies for its attributes.
    You can use these condition proxies to filter S2T models in thinkthinksyn backend,
    e.g.
    ```python
    async def test():
        return (await tts.transcription(
            audio=...,
            model_filter=(S2TModel.Name == WhisperV3Large.Name)
            ...
        ))['text']
    ```
    '''
    AvailableLangs = ConditionProxy[tuple[str, ...]|None, str]('AvailableLangs')
    '''
    languages that the model can recognize(iso639-1 codes, e.g. 'en' for English, 'zh' for Chinese, 'yue' for Cantonese, etc.).
    None for no language limitation.
    '''
    MaxAudioDurationSeconds = ConditionProxy[float|int, Never]('MaxAudioDurationSeconds')
    '''
    maximum audio duration(seconds) that the model can process in a single inference.
    Larger audio will be split into multiple segments automatically.
    '''
    PreferredSamplingRate = ConditionProxy[int|None, Never]('PreferredSamplingRate')
    '''preferred audio sampling rate(hz) for the model. None for no preference.'''
    PreferredSamplingWidth = ConditionProxy[int|None, Never]('PreferredSamplingWidth')
    '''preferred audio sampling width(bits) for the model. None for no preference.'''

StreamableAudioFormat: TypeAlias = Literal["wav", "opus", "aac", "mp3"]
'''Supported streamable audio formats. Note that this is a subset of `AudioFormat`'''

class T2SModel(_AIModel, type='t2s'):
    '''
    Text-to-Speech model data type with condition proxies for its attributes.
    You can use these condition proxies to filter T2S models in thinkthinksyn backend,
    e.g.
    ```python
    async def test():
        return (await tts.synthesis(
            text='hello world',
            model_filter=(T2SModel.Name == CosyVoiceV2.Name)
            ...
        ))['data']  # return base64-encoded audio data string
    ```
    '''
    AvailableLangs = ConditionProxy[tuple[str, ...]|None, str]('AvailableLangs')
    '''
    languages that the model can synthesize(iso639-1 codes, e.g. 'en' for English, 'zh' for Chinese, 'yue' for Cantonese, etc.).
    None for no language limitation.
    '''
    AvailableSpeakers = ConditionProxy[tuple[str, ...]|None, str]('AvailableSpeakers')
    '''speakers that the model can synthesize(str). None for no speaker limitation.'''
    MaxTextLen = ConditionProxy[int|None, Never]('MaxTextLen')
    '''The maximum length of the text that can be processed by this T2S model. Default is None(no limitation).'''
    SampleRate = ConditionProxy[int, Never]('SampleRate')
    '''The sample rate of the audio generated by this T2S model. Default is 24000.'''
    SampleWidth = ConditionProxy[int, Never]('SampleWidth')
    '''The sample width of the audio generated by this T2S model(in bytes). Default is 2(16 bits).'''
    Channels = ConditionProxy[int, Never]('Channels')
    '''The channel of the audio generated by this T2S model. Default is 1.'''
    OutputFormat = ConditionProxy[StreamableAudioFormat, Never]('OutputFormat')
    '''The audio format of the audio generated by this T2S model. Default is 'wav'.'''
    
class Img2TxtModel(_AIModel, type='img2txt'):
    '''
    Image-to-Text model data type with condition proxies for its attributes.
    You can use these condition proxies to filter Img2Txt models in thinkthinksyn backend,
    e.g.
    ```python
    async def test():
        return (await tts.image_captioning(
            image=...,
            model_filter=(Img2TxtModel.Name == SomeModel.Name)
            ...
        ))['text']
    ```
    '''

@dataclass(frozen=True)
class SpecialToken:
    """A special token for chat prompt.
    It includes the token string and the escape string for it."""

    token: str
    """the origin token string. Input token will be stripped."""
    escape: str
    """the string for escaping this token from prompt. 
    Default to be the token string which replaced `|` with `||`."""
    format: str = "%s"
    """format for replacing the token from string. Default to be `%s`.
    You must include `%s` in the format string."""
    _escape_escape: str = ""
    """the escape string for escaping the escape string itself."""

    def __init__(self, token: str, escape: str | None = None, format: str = "%s"):
        token = token.strip()
        if not token:
            raise ValueError(f"Invalid token: {token}")
        object.__setattr__(self, "token", token)
        if not escape:
            if "|" in token:
                escape = token.replace("|", "||")
            else:
                if len(token) > 1:
                    escape = "\\" + token[:-1] + "\\" + token[-1]
                else:
                    escape = "\\" + token
        if len(escape) > 1:
            escape_escape = "\\" + escape[:-1] + "\\" + escape[-1]
        else:
            escape_escape = "\\" + escape

        object.__setattr__(self, "_escape_escape", escape_escape)
        object.__setattr__(self, "escape", escape)
        object.__setattr__(self, "format", format)

    def __str__(self):
        return self.token

    def __repr__(self):
        return f"<SpecialToken: {self.token}>"

    def __eq__(self, other) -> bool:
        if isinstance(other, SpecialToken):
            return self.token == other.token and self.escape == other.escape and self.format == other.format
        elif isinstance(other, str):
            return self.token == other
        return False

    def __hash__(self):
        return hash(self.token + self.escape + self.format)

    def __add__(self, other) -> str:
        if isinstance(other, SpecialToken):
            return self.token + other.token
        if isinstance(other, str):
            return self.token + other
        raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __mod__(self, other) -> str:
        return self.token % other

    def __len__(self):
        return len(self.token)

    def __serialize__(self):
        # a special method for type_utils.serialize
        return self.token

    def escape_text(self, text: str, escaper: Callable[[str, "SpecialToken"], str] | None = None) -> str:
        """
        Escape the origin token in the given text, with the escape string, e.g. <|some token|> -> <||some token||>
        This is to prevent prompt injection from the input of user.

        You can provide a custom escaper through `escaper` param.
        """
        if escaper:
            return escaper(text, self)  # use the custom escaper function
        else:
            text = text.replace(self.escape, self._escape_escape)  # escape the escape string itself
            return text.replace(self.token, self.escape)

    def unescape_text(self, text: str, replace: str | None = None):
        """
        If escaped form of this token exist in the given text, replace them all with value in `replace` param.
        If `replace` is given, it will replace the token with `replace` instead of the original token
        """
        replace_str = replace if replace is not None else self.token
        text = text.replace(self.escape, replace_str)
        return text.replace(self._escape_escape, self.escape)  # unescape the escape string itself

    def formats(self, text: str, replacer: Union["SpecialToken", str]):
        """
        Replace the origin tokens in the given text with your given words.
        `format` will be used for the replacement value.
        """
        if isinstance(replacer, SpecialToken):
            replace_str = replacer.format % replacer.token
        elif isinstance(replacer, str):
            replace_str = replacer
        else:
            raise ValueError(f"Unsupported type for replacer: {type(replacer)}")
        return text.replace(self.token, self.format % replace_str)

__all__ = ['LLM', 'EmbeddingModel', 'S2TModel', 'T2SModel', 'Img2TxtModel']
# endregion bases

# region embedding models
class ZPointLarge(EmbeddingModel):
    '''The default embedding model provided by thinkthinksyn.'''
    Name = 'iampanda/zpoint_large_embedding_zh'
    Alias = ('zpoint',)
    DefaultEmbeddingDim = 1792

EmbeddingModel.RegisterDefault(ZPointLarge)

__all__.extend(['ZPointLarge'])
# endregion

# region reranking models
class _CommonRerankingModel(RerankingModel):
    MaxTokens: ClassVar[int|None] = None

class BGERerankerLarge(_CommonRerankingModel):
    '''The default reranking model provided by thinkthinksyn.'''
    Name: ClassVar[str] = 'BAAI/bge-reranker-large'
    Alias: ClassVar[tuple[str, ...]] = ('bge', 'bgeReranker', 'bge-reranker')
    MaxTokens: ClassVar[int] = 512

__all__.extend(['BGERerankerLarge'])
# endregion

# region s2t models
class _CommonS2TModel(S2TModel):
    PreferredSamplingRate = 16000
    PreferredSamplingWidth = 2

class WhisperV3Large(_CommonS2TModel):
    '''The default speech-to-text model provided by thinkthinksyn.'''
    Name = 'whisper'
    Alias = ('whisper', 'whisperv3', 'fast-whisper', 'whisper-large', 'whisper-v3')
    MaxAudioDurationSeconds = 60
    AvailableLangs = ('af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 
                        'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'haw', 'ha', 'he', 'hi', 'hr', 'ht', 
                        'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 
                        'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 
                        'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 
                        'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'yue', 'zh')

__all__.extend(['WhisperV3Large'])
# endregion

# region t2s models
class _CommonT2SModel(T2SModel):
    DefaultLang = None
    DefaultSpeaker = None
    MaxTextLen = None
    SampleRate = 24000
    SampleWidth = 2
    Channels = 1
    OutputFormat = 'wav'

class XttsV2(_CommonT2SModel):
    Name = 'xtts'
    Alias = ('xtts_v2',)
    AvailableLangs = (
        'en', 'es', 'fr', 'de', 'it', 'pt', 'pl',
        'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu',
        'ko', 'ja', 'hi'
    )
    AvailableSpeakers = (
        'Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence',
        'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 
        'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler',
        'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black',
        'Gilberto Mathias','Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid',
        'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 
        'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 
        'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 
        'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Ferran Simen', 
        'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski'
    )
    
class CosyVoiceV2(_CommonT2SModel):
    '''
    CosyVoice V2 from Alibaba TongYi. repo: https://github.com/FunAudioLLM/CosyVoice
    Supports passing instruction prompt to specify language/attitude/..., 
    Also supports passing audio prompt to do voice cloning.
    
    For more prompting examples, see: https://funaudiollm.github.io/cosyvoice2/
    
    NOTE: 
        1. There are no `speaker` in CosyVoice. Voice is determined by the audio prompt.
        2. if no audio prompt is specified, the default speaker voice will be used,
            which is bad in Cantonese. You should provide an Cantonese audio prompt
            in that case.
        3. special tags are allowed in `text`. See `CosyVoiceSpecialTag` for details.
    '''
    AvailableLangs = (
        'en', 'zh-cn', 'zh-tw', 'ja', 'ko', 'yue', 
        'sichuan', 'shanghai', 'tianjin', 'changsha', 'zhengzhou'
    )
    SampleRate = 24000

__all__.extend(['XttsV2', 'CosyVoiceV2'])
# endregion

# region LLM
LLM_DEFAULT_USER_ROLE_NAME = "user"
LLM_DEFAULT_ASSIST_ROLE_NAME = "assistant"
LLM_DEFAULT_SYSTEM_ROLE_NAME = "system"

LLM_DEFAULT_SYSTEM_ROLE_ALIAS: Final[tuple[str, ...]] = ('system', 'sys', 'admin', 'root', 'superuser', 'super')
LLM_DEFAULT_USER_ROLE_ALIAS:Final[tuple[str, ...]] = ('user', 'users', 'human', 'person', 'you', 'me', 'I')
LLM_DEFAULT_ASSIST_ROLE_ALIAS:Final[tuple[str, ...]] = ('assistant', 'assist', 'bot', 'robot', 'ai', 'gpt', 'gpt4', 'helper', 
                                                    'model', 'starling', 'gemma', 'qwen', 'llama')

@dataclass(frozen=True)
class LLMChatRole:
    """The data class for containing necessary information of a role in a chat model. E.g. user, assistant, system."""

    name: str
    """the correct name of the role. E.g. 'user', 'assistant', 'system'."""
    alias: CaseIgnoreStrSet
    """the alias of the role. E.g. ['ai', 'bot', 'helper', 'robot']. `name` will also be included in this set after the initialization."""
    format: str = "%s"
    """the format for replacing the placeholder with the role's correct name. Default is '%s'. 
    E.g. `%s :` -> `user :` """

    def __str__(self):
        return self.name

    def __init__(self, name: str, alias: str | Sequence[str] | None = None, format: str = "%s"):
        object.__setattr__(self, "name", name.strip())

        if not alias:
            alias = []
        else:
            alias = ([alias,] if isinstance(alias, str) else list(alias))
        alias_set = CaseIgnoreStrSet([self.name,] + [a.strip() for a in alias])
        object.__setattr__(self, "alias", alias_set)
        object.__setattr__(self, "format", format)

    def __eq__(self, other) -> bool:
        if isinstance(other, LLMChatRole):
            return self.name == other.name and self.alias == other.alias and self.format == other.format
        elif isinstance(other, str):
            return self.name == other or \
                self.alias.contains(other) or \
                    self.formatted_name == other
        return False

    def __repr__(self):
        return f"ChatRole(name=`{self.name}`)"

    @property
    def formatted_name(self):
        return self.format % self.name

__all__.extend([
    'LLM_DEFAULT_USER_ROLE_NAME', 'LLM_DEFAULT_ASSIST_ROLE_NAME', 'LLM_DEFAULT_SYSTEM_ROLE_NAME',
    'LLM_DEFAULT_SYSTEM_ROLE_ALIAS', 'LLM_DEFAULT_USER_ROLE_ALIAS', 'LLM_DEFAULT_ASSIST_ROLE_ALIAS',
    'LLMChatRole'
])

class _CommonLLM(LLM):
    MoE: ClassVar[bool] = False
    SupportsImage: ClassVar[bool] = False
    SupportsAudio: ClassVar[bool] = False
    SupportsVideo: ClassVar[bool] = False
    
    DefaultUserRole: ClassVar[LLMChatRole] = LLMChatRole(LLM_DEFAULT_USER_ROLE_NAME, alias=LLM_DEFAULT_USER_ROLE_ALIAS)
    DefaultAssistRole: ClassVar[LLMChatRole] = LLMChatRole(LLM_DEFAULT_ASSIST_ROLE_NAME, alias=LLM_DEFAULT_ASSIST_ROLE_ALIAS)
    DefaultSystemRole: ClassVar[LLMChatRole] = LLMChatRole(LLM_DEFAULT_SYSTEM_ROLE_NAME, alias=LLM_DEFAULT_SYSTEM_ROLE_ALIAS)
    
    DefaultSystemPrompt: ClassVar[str|None] = None
    PromptStartToken: ClassVar[SpecialToken | None] = None
    ChatStartToken: ClassVar[SpecialToken | None] = None
    ChatContentStartToken: ClassVar[SpecialToken | None] = None
    ChatEndToken: ClassVar[SpecialToken | None] = None
    StopTokens: ClassVar[set[str]] = None  # type: ignore
    
# region qwen
class Qwen3_30B_A3B(_CommonLLM):
    '''Non-instruction version of Qwen3-30B-A3B, i.e. model with thinking (<think>) capability.'''
    Name: ClassVar[str] = 'Qwen/Qwen3-30B-A3B'
    B: ClassVar[int] = 30
    MoE: ClassVar[bool] = True
    Alias: ClassVar[tuple[str, ...]] = ('qwen', 'qwen3', 'qwen3-30b', 'qwen3_30b', 'qwen3-30b-non-instruct')
    
    DefaultSystemPrompt: ClassVar[str] = 'You are a helpful assistant. '
    
    DefaultUserRole: ClassVar[LLMChatRole] = LLMChatRole(LLM_DEFAULT_USER_ROLE_NAME, alias=LLM_DEFAULT_USER_ROLE_ALIAS, format='%s\n')
    DefaultAssistRole: ClassVar[LLMChatRole] = LLMChatRole(LLM_DEFAULT_ASSIST_ROLE_NAME, alias=LLM_DEFAULT_ASSIST_ROLE_ALIAS, format='%s\n')
    DefaultSystemRole: ClassVar[LLMChatRole] = LLMChatRole(LLM_DEFAULT_SYSTEM_ROLE_NAME, alias=LLM_DEFAULT_SYSTEM_ROLE_ALIAS, format='%s\n')
    
    ChatStartToken: ClassVar[SpecialToken] = SpecialToken('<|im_start|>')
    ChatEndToken: ClassVar[SpecialToken] = SpecialToken('<|im_end|>', format='%s\n')
    StopTokens: ClassVar[set[str]] = {'<|im_end|>', '<|endoftext|>'}
    
class Qwen3_30B_A3B_Instruct(Qwen3_30B_A3B):
    '''Instruction version of Qwen3-30B-A3B.'''
    Name: ClassVar[str] = 'Qwen/Qwen3-30B-A3B-Instruct'
    Alias: ClassVar[tuple[str, ...]] = ('qwen', 'qwen3', 'qwen3-30b', 'qwen3_30b', 'qwen3-30b-instruct')
    
class Qwen3_30B_A3B_Omni_Instruct(Qwen3_30B_A3B):
    '''Multimodal instruction version of Qwen3-30B-A3B.'''
    Name: ClassVar[str] = 'Qwen/Qwen3-Omni-30B-A3B-Instruct'
    Alias: ClassVar[tuple[str, ...]] = ('qwen', 'qwen3', 'qwen3-30b', 'qwen3_30b', 'qwen3-omni-instruct', 'omni', 'qwen3-omni', 'qwen3-30b-omni')
    SupportsImage: ClassVar[bool] = True
    SupportsAudio: ClassVar[bool] = True
    SupportsVideo: ClassVar[bool] = True
    
__all__.extend([
    'Qwen3_30B_A3B', 'Qwen3_30B_A3B_Instruct', 'Qwen3_30B_A3B_Omni_Instruct'
])
# endregion qwen

# region gemma
class Gemma2_9B_Instruct(_CommonLLM):
    Name = 'google/gemma-2-9b-it'
    Alias = ('gemma', 'gemma2', 'gemma2-9B')
    B = 9
    
    # no system prompt by default for gemma2
    
    DefaultUserRole = LLMChatRole(LLM_DEFAULT_USER_ROLE_NAME, alias=LLM_DEFAULT_USER_ROLE_ALIAS, format='%s\n')
    DefaultAssistRole = LLMChatRole('model', alias=LLM_DEFAULT_ASSIST_ROLE_ALIAS, format='%s\n')
    DefaultSystemRole = LLMChatRole(LLM_DEFAULT_USER_ROLE_NAME, alias=LLM_DEFAULT_SYSTEM_ROLE_ALIAS, format='%s\n')    
    # gemma2 has not trained with system role, always use `user` as system role.
    
    ChatStartToken = SpecialToken('<start_of_turn>')
    ChatEndToken = SpecialToken('<end_of_turn>', format='%s\n')

class Gemma3_27B_Instruct(Gemma2_9B_Instruct):
    Name = 'google/gemma-3-27b-it'
    Alias = ('gemma', 'gemma3', 'gemma3-27B')
    B = 27
    SupportsImage = True

__all__.extend([
    'Gemma2_9B_Instruct', 'Gemma3_27B_Instruct'
])
# endregion gemma
# endregion


if __name__ == "__main__":  # for debugging
    print(ZPointLarge.ServiceType)
    print(ZPointLarge.ClassName)