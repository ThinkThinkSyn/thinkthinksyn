from typing import Never, TypeAlias, Literal, Type, Self, TypeVar, Generic
from .base import ConditionProxy

# region bases
class _AIModel:
    def __new__(cls):
        raise TypeError(f"{cls.__name__} is a data type class and cannot be instantiated.")
    
    Name = ConditionProxy[str, Never]('Name')
    '''model name registered in thinkthinksyn backend'''
    Alias = ConditionProxy[tuple[str, ...], str]('Alias')
    '''model alias, i.e. other names for the model'''

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

class LLM(_AIModel): 
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

class _ChildEmbeddingModel(_ChildAIModel):
    DefaultEmbeddingDim: int

class EmbeddingModel(_DefaultableAIModel[_ChildEmbeddingModel]):
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
    
class S2TModel(_AIModel):
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

class T2SModel(_AIModel):
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
    
class Img2TxtModel(_AIModel):
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
class _CommonLLM(LLM):
    MoE = False
    SupportsImage = False
    SupportsAudio = False
    SupportsVideo = False
    
# region qwen
class Qwen3_30B_A3B(_CommonLLM):
    '''Non-instruction version of Qwen3-30B-A3B, i.e. model with thinking (<think>) capability.'''
    Name = 'Qwen/Qwen3-30B-A3B'
    B = 30
    MoE = True
    Alias = ('qwen', 'qwen3', 'qwen3-30b', 'qwen3_30b', 'qwen3-30b-non-instruct')
        
class Qwen3_30B_A3B_Instruct(Qwen3_30B_A3B):
    '''Instruction version of Qwen3-30B-A3B.'''
    Name = 'Qwen/Qwen3-30B-A3B-Instruct'
    Alias = ('qwen', 'qwen3', 'qwen3-30b', 'qwen3_30b', 'qwen3-30b-instruct')
    
class Qwen3_30B_A3B_Omni_Instruct(Qwen3_30B_A3B):
    '''Multimodal instruction version of Qwen3-30B-A3B.'''
    Name = 'Qwen/Qwen3-Omni-30B-A3B-Instruct'
    Alias = ('qwen', 'qwen3', 'qwen3-30b', 'qwen3_30b', 'qwen3-omni-instruct', 'omni', 'qwen3-omni', 'qwen3-30b-omni')
    SupportsImage = True
    SupportsAudio = True
    SupportsVideo = True
    
__all__.extend([
    'Qwen3_30B_A3B', 'Qwen3_30B_A3B_Instruct', 'Qwen3_30B_A3B_Omni_Instruct'
])
# endregion qwen

# region gemma
class Gemma2_9B_Instruct(_CommonLLM):
    Name = 'google/gemma-2-9b-it'
    Alias = ('gemma', 'gemma2', 'gemma2-9B')
    B = 9

class Gemma3_27B_Instruct(_CommonLLM):
    Name = 'google/gemma-3-27b-it'
    Alias = ('gemma', 'gemma3', 'gemma3-27B')
    B = 27
    SupportsImage = True

__all__.extend([
    'Gemma2_9B_Instruct', 'Gemma3_27B_Instruct'
])
# endregion gemma
# endregion