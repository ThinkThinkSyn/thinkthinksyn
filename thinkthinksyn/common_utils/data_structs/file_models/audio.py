import requests

from io import BytesIO
from pathlib import Path
from pydub import AudioSegment
from pydantic_core import core_schema
from pydub.silence import split_on_silence
from pydub.utils import audioop, ratio_to_db
from typing import cast, Self, TYPE_CHECKING, override, Generator, Coroutine, Literal, TypeAlias

from ...type_utils import bytes_to_base64, base64_to_bytes
from ...concurrent_utils import run_any_func, is_async_callable
from .image import _hash_md5

AudioFormat: TypeAlias = Literal["wav", "mp3", "aac", "flac", "opus", "ogg", "m4a", "wma"]
'''Supported non-stream response audio formats.'''

StreamableAudioFormat: TypeAlias = Literal["wav", "opus", "aac", "mp3"]
'''Supported streamable audio formats. Note that this is a subset of `AudioFormat`'''

class Audio(AudioSegment):
    '''
    Advance audio model for easy validation in pydantic.
    Available deserialization formats:
     - Path: the path to the audio file
     - str: the path to the audio file/ base64 string of the audio (detect by `/` in the string)
     - bytes: the bytes of the audio
     - AudioSegment: the audio segment object
     - dict: the dict with key `voice`/`sound`/`audio`/`data`/`source`/`url` (detect by the key in the dict),
             or dict data with these keys:
                - `format`(must provide, e.g. wav, mp3, ...)
                - `frame_rate`(must provide)
                - `channels`(default=1)
                - `sample_width`(default=2)
                - `data`: the audio data in bytes/base64 format/path
    '''
    
    start_time: float = 0.0
    _end_time: float|None = None
    
    @property
    def end_time(self)->float:
        return self._end_time if self._end_time is not None else self.duration_seconds
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validator(data):
            if isinstance(data, dict):
                if ('voice' in data) or ('sound' in data) or ('audio' in data) or ('data' in data) or ('source' in data) or ('url' in data):
                    audio = data.get('voice', data.get('sound', data.get('audio', data.get('data', data.get('source', data.get('url'))))))
                    if audio and isinstance(audio, (Path, str, bytes, AudioSegment)):
                        data = audio # continue to the next step
                elif 'format' in data or 'Format' in data:
                    format = data.get('format', data.get('Format', None))
                    frame_rate = data.get('frame_rate', data.get('FrameRate', None))
                    channels = data.get('channels', data.get('Channels', 1))
                    sample_width = data.get('sample_width', data.get('SampleWidth', 2))
                    audio_data = data.get('data', data.get('Data', data.get('source', data.get('Source', data.get('url', data.get('URL', None))))))
                    
                    if not frame_rate or not format or not audio_data:
                        raise ValueError('Invalid audio data. Must provide `format`, `frame_rate`, and `data`')
                    if isinstance(audio_data, str):
                        if '/' in audio_data or '\\' in audio_data:
                            if audio_data.startswith('http'):
                                audio_data = requests.get(audio_data).content
                            else:
                                audio_data = Path(audio_data)
                        else:   # assume base64 string1
                            audio_data = base64_to_bytes(audio_data)
                    if isinstance(audio_data, Path):
                        if not audio_data.exists():
                            raise ValueError(f'Invalid path: {audio_data}')
                        audio_data = audio_data.read_bytes()
                    if not isinstance(audio_data, bytes):
                        raise ValueError('Invalid audio data. Must be bytes or base64 string.')
                    audio_data = BytesIO(audio_data)
                    channels = int(channels) if channels else 1
                    sample_width = int(sample_width) if sample_width else 2
                    data = AudioSegment.from_file(audio_data, format=format, frame_rate=frame_rate, channels=channels, sample_width=sample_width)
                    
            if isinstance(data, (Path, str, bytes, AudioSegment)):
                data = cls.Load(data)
            return data
        
        def serializer(audio: 'Audio'):
            return audio.to_base64()

        validate_schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        serialize_schema = core_schema.plain_serializer_function_ser_schema(serializer)
        return core_schema.json_or_python_schema(
            json_schema=validate_schema,
            python_schema=validate_schema,
            serialization=serialize_schema
        )
    
    def __setattr__(self, name, val):
        if name == '_data' and hasattr(self, '_data'):
            if val != self._data:
                self.__md5_cache__ = {}
        super().__setattr__(name, val)
    
    @property
    def frame_size(self)->int:
        '''Frame size of the audio, in bytes.'''
        return self.frame_rate * self.frame_width  # type: ignore
    
    def to_bytes(self, format: str = 'wav')-> bytes:
        '''
        Get the data of this audio in bytes format.
        You can specify the output format of the audio bytes.
        '''
        buffer = BytesIO()
        self.export(buffer, format=format)
        return buffer.getvalue()
    
    def copy(self)->Self:
        '''Copy the audio object.'''
        audio = AudioSegment(self._data, sample_width=self.sample_width, frame_rate=self.frame_rate, channels=self.channels)
        return self.Load(audio) # type: ignore
    
    def to_base64(self, format: str='wav', url_scheme: bool=False)->str:
        '''
        Get the data of this audio in base64 format.
        Args:
            - format: the output format of the audio bytes, default is 'wav'.
            - url_scheme: if True, the base64 string will be prefixed with 'data:audio/{format};base64,',
                            which can be used directly in HTML audio tags.
        '''
        b64 = bytes_to_base64(self.to_bytes(format=format))
        if url_scheme:
            return f'data:audio/{format.lower()};base64,{b64}'
        return b64

    def to_md5_hash(self, format: str='wav')->str:
        '''
        Export the audio data to bytes and get the md5 hash
        of the audio data.
        '''
        if '__md5_cache__' not in self.__dict__:
            self.__md5_cache__ = {}
        if format not in self.__md5_cache__:
            self.__md5_cache__[format] = _hash_md5(self.to_bytes(format=format))
        return self.__md5_cache__[format]
    
    @property
    def min_dBFS(self):
        min_sound, _ = audioop.minmax(self._data, self.sample_width)    # type: ignore
        min_sound = (min_sound + self.max_possible_amplitude) / 2
        return ratio_to_db(min_sound, self.max_possible_amplitude)
    
    def split_on_silence(
        self, 
        min_silence_len: int = 500,
        silence_threshold: int|None = None,
        keep_silence: int|bool = 100, 
        seek_step: int = 1
    )->list['Audio']:
        '''
        Split the audio on silence.

        Args:
            min_silence_len: (in ms) minimum length of a silence to be used for
                            a split. default: 500ms

            silence_thresh - (in dBFS) anything quieter than this will be 
                             considered silence. If None, it will be set to 2 x self.dBFS.

            keep_silence: (in ms or True/False) leave some silence at the beginning
                            and end of the chunks. Keeps the sound from sounding like it
                            is abruptly cut off.
                            When the length of the silence is less than the keep_silence duration
                            it is split evenly between the preceding and following non-silent
                            segments.
                            If True is specified, all the silence is kept, if False none is kept.
                            default: 100ms
            seek_step - step size for interacting over the segment in ms
        '''
        silence_threshold_int: int = silence_threshold if silence_threshold is not None else (2 * self.dBFS)    # type: ignore
        if silence_threshold_int == -float("infinity"):
            silence_threshold_int = -32     # default value
        segs = split_on_silence(self, min_silence_len=min_silence_len, silence_thresh=silence_threshold_int, keep_silence=keep_silence, seek_step=seek_step)
        return [Audio.CastAudio(seg) for seg in segs]   # type: ignore
    
    def reduce_noise(
        self, 
        stationary: bool=False,
        prop_decrease: float=1.0,
        time_constant_s: float=2.0,
        freq_mask_smooth_hz: int=500,
        time_mask_smooth_ms: int=50,
        thresh_n_mult_nonstationary: int=2,
        sigmoid_slope_nonstationary: int=10,
        n_std_thresh_stationary: float=1.5,
        chunk_size: int=600000,
        padding: int=30000,
        n_fft: int=1024,
        win_length: int|None=None,
        hop_length: int|None=None
    )->Self:
        '''
        Reduce noise in the audio(return a new audio object)
        
        Args:
            stationary : bool, optional
                Whether to perform stationary, or non-stationary noise reduction, by default False
            prop_decrease : float, optional
                The proportion to reduce the noise by (1.0 = 100%), by default 1.0
            time_constant_s : float, optional
                The time constant, in seconds, to compute the noise floor in the non-stationary
                algorithm, by default 2.0
            freq_mask_smooth_hz : int, optional
                The frequency range to smooth the mask over in Hz, by default 500
            time_mask_smooth_ms : int, optional
                The time range to smooth the mask over in milliseconds, by default 50
            thresh_n_mult_nonstationary : int, optional
                Only used in nonstationary noise reduction., by default 1
            sigmoid_slope_nonstationary : int, optional
                Only used in nonstationary noise reduction., by default 10
            n_std_thresh_stationary : int, optional
                Number of standard deviations above mean to place the threshold between
                signal and noise., by default 1.5
            chunk_size : int, optional
                Size of signal chunks to reduce noise over. Larger sizes
                will take more space in memory, smaller sizes can take longer to compute.
                , by default 60000
                padding : int, optional
                How much to pad each chunk of signal by. Larger pads are
                needed for larger time constants., by default 30000
            n_fft : int, optional
                length of the windowed signal after padding with zeros.
                The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
                The default value, ``n_fft=2048`` samples, corresponds to a physical
                duration of 93 milliseconds at a sample rate of 22050 Hz.
                This value is well adapted for music signals. However, in speech processing, the recommended value is 512,
                corresponding to 23 milliseconds at a sample rate of 22050 Hz.
                In any case, we recommend setting ``n_fft`` to a power of two for
                optimizing the speed of the fast Fourier transform (FFT) algorithm., by default 1024
            win_length : int, optional
                Each frame of audio is windowed by ``window`` of length ``win_length``
                and then padded with zeros to match ``n_fft``.
                Smaller values improve the temporal resolution of the STFT (i.e. the
                ability to discriminate impulses that are closely spaced in time)
                at the expense of frequency resolution (i.e. the ability to discriminate
                pure tones that are closely spaced in frequency). This effect is known
                as the time-frequency localization trade-off and needs to be adjusted
                according to the properties of the input signal ``y``.
                If unspecified, defaults to ``win_length = n_fft``., by default None
            hop_length : int, optional
                number of audio samples between adjacent STFT columns.
                Smaller values increase the number of columns in ``D`` without
                affecting the frequency resolution of the STFT.
                If unspecified, defaults to ``win_length // 4`` (see below)., by default None
        '''
        audio_arr = self.get_array_of_samples()
        try:
            from noisereduce import reduce_noise as _reduce_noise
        except ImportError:
            raise ImportError('noisereduce package is required for noise reduction. Please install it via pip install thinkthinksyn[full] or `pip install noisereduce`')
        reduced_noise = _reduce_noise(
            audio_arr, 
            self.frame_rate, 
            stationary=stationary,
            prop_decrease=prop_decrease,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
            sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
            chunk_size=chunk_size,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length
        )    
        new_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=self.frame_rate,
            sample_width=self.sample_width,
            channels=self.channels
        )
        return Audio.CastAudio(new_audio)  # type: ignore
    
    @override
    def append(
        self, 
        seg: AudioSegment|bytes|str, 
        crossfade: int = 100,
        noise_reduce: bool=False,
        adjust_dBFS: bool=False,
    )->Self:
        '''
        Append another audio segment to this audio segment(return a new audio object)
        
        Args:
            seg: the audio segment to append
            crossfade: the length of the crossfade in milliseconds
            noise_reduce: whether to reduce noise in the audio to be appended
            adjust_dBFS: whether to adjust the dBFS of the audio to be appended,
                        so that the average dBFS of the appended audio is the same as
                        the average dBFS of this audio.
        '''
        if isinstance(seg, Audio):
            new_start = min(self.start_time, seg.start_time)    # type: ignore
            new_end = max(self.end_time, seg.end_time)  # type: ignore
        else:
            new_start = new_end = None
        seg = Audio.Load(seg)
        
        if noise_reduce:
            seg = seg.reduce_noise()
            
        if adjust_dBFS:
            origin_audio_avg_dBFS = self.dBFS
            append_audio_avg_dBFS = seg.dBFS
            diff = origin_audio_avg_dBFS - append_audio_avg_dBFS
            if diff != 0:
                seg = seg.apply_gain(diff)
        
        new_audio = super().append(seg, crossfade=crossfade)
        if new_start is not None and new_end is not None:
            new_audio.start_time = new_start
            new_audio._end_time = new_end
            
        return self.CastAudio(new_audio)    # type: ignore
    
    def prepend(
        self, 
        seg: AudioSegment|bytes|str,
        crossfade: int = 100,
        noise_reduce: bool=False,
        adjust_dBFS: bool=False,
    )->Self:
        '''
        Prepend another audio segment to this audio segment(return a new audio object)
        
        Args:
            seg: the audio segment to prepend
            crossfade: the length of the crossfade in milliseconds
            noise_reduce: whether to reduce noise in the audio to be prepended
            adjust_dBFS: whether to adjust the dBFS of the audio to be prepended,
                        so that the average dBFS of the prepended audio is the same as
                        the average dBFS of this audio.
        '''
        if isinstance(seg, Audio):
            new_start = min(self.start_time, seg.start_time)    # type: ignore
            new_end = max(self.end_time, seg.end_time)
        else:
            new_start = new_end = None
        seg = Audio.Load(seg)
        
        if noise_reduce:
            seg = seg.reduce_noise()
            
        if adjust_dBFS:
            origin_audio_avg_dBFS = self.dBFS
            prepend_audio_avg_dBFS = seg.dBFS
            diff = origin_audio_avg_dBFS - prepend_audio_avg_dBFS
            if diff != 0:
                seg = seg.apply_gain(diff)
                
        new_audio = AudioSegment.append(seg, self, crossfade=crossfade)
        if new_start is not None and new_end is not None:
            new_audio.start_time = new_start
            new_audio._end_time = new_end
            
        return self.CastAudio(new_audio)    # type: ignore
    
    @classmethod
    def Load(cls, data: str|bytes|AudioSegment|Path, /)->Self:
        '''Load audio from data. If the data is already an AudioSegment, it will be casted to this class.'''
        if isinstance(data, (str, bytes, Path)):
            data = run_any_func(_get_audio, data)
            if data is None:
                raise ValueError('Invalid audio data')
        if not isinstance(data, cls) and isinstance(data, AudioSegment):
            data = cls.CastAudio(data)
        return data # type: ignore

    @classmethod
    async def ALoad(cls, data: str|bytes|AudioSegment|Path)->Self:
        '''Asynchronously load audio from data. If the data is already an AudioSegment, it will be casted to this class.'''
        if isinstance(data, (str, bytes, Path)):
            data = await run_any_func(_get_audio, data)
            if data is None:
                raise ValueError('Invalid audio data')
        if not isinstance(data, cls) and isinstance(data, AudioSegment):
            data = cls.CastAudio(data)
        return data # type: ignore

    @classmethod
    def CastAudio(cls, audio: AudioSegment)->Self:   
        '''change origin audio type(AudioSegment) to this advance audio type'''
        if isinstance(audio, cls):
            return audio
        setattr(audio, '__class__', cls)
        audio = cast(cls, audio)    # type: ignore
        return audio  # type: ignore
    
    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} duration={self.duration_seconds}s>'
    
    def __getitem__(self, millisecond: slice|int)->list[Self]|Self:
        r = AudioSegment.__getitem__(self, millisecond)
        if isinstance(r, Generator):    # type: ignore
            return [Audio.CastAudio(seg) for seg in r]  # type: ignore
        else:
            if isinstance(millisecond, slice):
                start = millisecond.start if millisecond.start is not None else 0
                end = millisecond.stop if millisecond.stop is not None else len(self)
                start = min(start, len(self))
                end = min(end, len(self))
            else:
                start = millisecond
                end = millisecond + 1
            audio = Audio.CastAudio(r)  # type: ignore
            audio.start_time = start / 1000   # convert to seconds
            audio._end_time = end / 1000
            return audio    # type: ignore
    
    if not TYPE_CHECKING:
        def __getattribute__(self, name: str):
            attr = super().__getattribute__(name)
            if not (name.startswith('__') and name.endswith('__')):
                if not isinstance(attr, type) and callable(attr):
                    return _AudioRetWrapper(attr)
            return attr

class _AudioRetWrapper:
    def __init__(self, f):
        self.f = f
        if hasattr(self.f, '__doc__'):
            self.__doc__ = self.f.__doc__
        
    def __getattr__(self, name: str):
        return getattr(self.f, name)
    
    @staticmethod
    def _recursive_cast_audio(r):
        if isinstance(r, AudioSegment) and not isinstance(r, Audio):
            return Audio.CastAudio(r)
        elif isinstance(r, (list, tuple, set)):
            return type(r)([_AudioRetWrapper._recursive_cast_audio(item) for item in r])
        elif isinstance(r, dict):
            return type(r)({key: _AudioRetWrapper._recursive_cast_audio(val) for key, val in r.items()})
        return r

    def __call__(self, *args, **kwargs):
        r = self.f(*args, **kwargs)
        if isinstance(r, Coroutine):
            async def wrapper():
                coro_r = await r
                return _AudioRetWrapper._recursive_cast_audio(coro_r)
            return wrapper()
        else:
            r = _AudioRetWrapper._recursive_cast_audio(r)
            return r
    
    def __is_async_func__(self)->bool:
        # `for `is_async_callable` to work
        return is_async_callable(self.f)
    

__all__ = ['Audio', 'AudioFormat', 'StreamableAudioFormat']