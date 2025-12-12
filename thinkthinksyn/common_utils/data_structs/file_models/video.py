if __name__ == "__main__":  # for debugging
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'common_utils.data_structs.file_models'

from ._utils import _init_ffmpeg
_init_ffmpeg()

import os
import base64
import tempfile
import numpy as np

from io import BytesIO
from pathlib import Path
from functools import partial
from pydantic_core import core_schema
from typing import Self, Callable, TYPE_CHECKING, TypeAlias, Literal

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader as MoviePyFFMPEGVideoReader
from moviepy.video.VideoClip import VideoClip

from ...concurrent_utils import run_any_func
from ...type_utils import check_value_is
from .loader import AcceptableFileSource, save_get
from ._utils import _try_get_from_dict, _get_media_json_schema, _dump_media_dict

if TYPE_CHECKING:
    from .audio import Audio
    from .image import Image

_tempfile_cls: TypeAlias = tempfile._TemporaryFileWrapper

class _FFMPEGTempFileVideoReader(MoviePyFFMPEGVideoReader):
    def __init__(self, file: str|_tempfile_cls, *args, **kwargs):
        self._temp_file = None
        if isinstance(file, _tempfile_cls):
            filename = file.name
            self._temp_file = file
        else:
            filename = file
        super().__init__(filename, *args, **kwargs)
    
    def close(self, delete_lastread=True):
        try:
            super().close(delete_lastread)
            if delete_lastread and self._temp_file:
                tmp = self._temp_file
                self._temp_file = None
                tmp.close()
        except:
            pass
            
_defer_attrs = ('duration', 'end', 'fps', 'size', 'rotation', 'frame_function',)

VideoCommonFormats: TypeAlias = Literal['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv']

_VideoTypeCodecMap = {
    'mp4': 'libx264',
    'avi': 'libx264',
    'mov': 'libx264',
    'mkv': 'libx264',
    'webm': 'libvpx-vp9',
    'flv': 'flv',
    'wmv': 'wmv2',
}

def _tidy_video_format(format: str|None, raise_err=True, default='mp4')->VideoCommonFormats:
    if not format:
        return default  # type: ignore
    format = format.lower().strip('. ')
    if format not in _VideoTypeCodecMap:
        if raise_err:
            raise ValueError(f"Unsupported video format: {format}")
        return format   # type: ignore
    return format   # type: ignore

class Video(VideoClip):
    
    duration: float
    '''The duration of the video clip in seconds.'''
    end: float
    '''The end time of the video clip in seconds.'''
    fps: float
    '''The frames per second of the video clip.'''
    size: tuple[int, int]
    '''The (width, height) of the video clip in pixels.'''
    rotation: int
    '''The rotation of the video clip in degrees.'''
    frame_function: Callable[[float], 'np.ndarray']
    '''A function that takes a time (in seconds) and returns the corresponding frame as a
    numpy array.'''
    
    mask: VideoClip|None = None
    '''A mask video clip for the video clip, if any.'''
    audio: AudioFileClip|None = None
    '''An audio clip for the video clip, if any.
    If you want to get `Audio` object, use `get_audio_model()` method.'''
    
    reader: _FFMPEGTempFileVideoReader|None = None
    '''
    A video clip object that loads video data from a file source.
    NOTE: The video data is loaded lazily when needed. `reader` will 
    be None until the video is actually used.
    '''
    _defer_loader = None
    _origin_format = None
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validator(data):
            if isinstance(data, dict):
                data = _try_get_from_dict(data, 'data', 'content', 'video', 'clip', 'source', 'url')
            if not isinstance(data, cls):
                data = cls.Load(data)   # type: ignore
            return data
        
        def serializer(video: 'Video'):
            return _dump_media_dict(video.to_base64(), cls)

        validate_schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        serialize_schema = core_schema.plain_serializer_function_ser_schema(serializer)
        return core_schema.json_or_python_schema(
            json_schema=validate_schema,
            python_schema=validate_schema,
            serialization=serialize_schema
        )
    
    @classmethod
    def __get_pydantic_json_schema__(cls, cs, handler):
        return _get_media_json_schema(cls)
    
    def _defer_init(
        self, 
        reader: _FFMPEGTempFileVideoReader, 
        has_mask: bool=False, 
        audio: bool=True,
        audio_fps: int=44100,
        audio_nbytes: int=2,
        audio_buffersize: int=200000,
    ):
        self.duration = reader.duration
        self.end = reader.duration
        self.fps = reader.fps
        self.size = tuple(reader.size)  # type: ignore
        self.rotation = reader.rotation
        
        if has_mask:
            self.frame_function = lambda t: reader.get_frame(t)[:, :, :3]

            def mask_frame_function(t):
                return reader.get_frame(t)[:, :, 3] / 255.0

            self.mask = VideoClip(is_mask=True, frame_function=mask_frame_function).with_duration(self.duration)
            self.mask.fps = self.fps    # type: ignore
        else:
            self.frame_function = lambda t: reader.get_frame(t)
            
        if audio and reader.infos["audio_found"]:
            self.audio = AudioFileClip(
                reader.filename,
                buffersize=audio_buffersize,
                fps=audio_fps,
                nbytes=audio_nbytes,
            )
        self._defer_loader = None
    
    def _create_defer_loader(
        self,
        source: AcceptableFileSource,
        decode_file:bool=False,
        has_mask: bool=False,
        audio: bool=True,
        audio_buffersize: int=200000,
        target_resolution=None,
        resize_algorithm="bicubic",
        audio_fps=44100,
        audio_nbytes: int=2,
        fps_source="fps",
        pixel_format=None,
    ):
        if isinstance(source, Path):
            source_str = str(source)
        elif isinstance(source, str):
            source_str = source
        else:
            source_str = None
        if source_str and len(source_str) < 1024 and ('.' in source_str[-5:]):
            suffix = source_str.split('.')[-1]
        else:
            suffix = None
        
        if suffix:
            try:
                maybe_format = _tidy_video_format(suffix, raise_err=True, default=None) # type: ignore
            except:
                maybe_format = None
        else:
            maybe_format = None
        self._origin_format = maybe_format
        
        def defer_loader(source):
            video_file = None
            if source_str and len(source_str) < 1024:   # seems like a path
                if not source_str.startswith(('http://', 'https://', 'ftp://', 's3://', 'gs://')):
                    if os.path.exists(source_str):
                        video_file = source_str
            if not video_file:
                video_file = tempfile.NamedTemporaryFile(delete=False, suffix=(f'.{suffix}' if suffix else ''), mode='wb')
                video_file.write(run_any_func(save_get, source).read())
                
            self.reader = reader = _FFMPEGTempFileVideoReader(
                video_file,
                decode_file=decode_file,
                pixel_format=pixel_format,  # type: ignore
                target_resolution=target_resolution,
                resize_algo=resize_algorithm,
                fps_source=fps_source,
            )
            self._defer_init(reader, has_mask=has_mask, audio=audio, audio_fps=audio_fps, 
                             audio_nbytes=audio_nbytes, audio_buffersize=audio_buffersize)
            
        return partial(defer_loader, source)

    def __init__(
        self,
        source: AcceptableFileSource,
        decode_file:bool=False,
        has_mask: bool=False,
        audio: bool=True,
        audio_buffersize: int=200000,
        target_resolution=None,
        resize_algorithm="bicubic",
        audio_fps=44100,
        audio_nbytes: int=2,
        fps_source="fps",
        pixel_format=None,
        is_mask: bool=False,
    ):
        VideoClip.__init__(self, is_mask=is_mask)

        # Make a reader
        if not pixel_format:
            pixel_format = "rgba" if has_mask else "rgb24"

        if not self.reader:
            self._defer_loader = self._create_defer_loader(
                source,
                decode_file=decode_file,
                pixel_format=pixel_format,
                target_resolution=target_resolution,
                resize_algorithm=resize_algorithm,
                fps_source=fps_source,
            )
            setattr(self._defer_loader, '__source__', source)
        else:
            self._defer_init(self.reader, has_mask=has_mask, audio=audio, audio_fps=audio_fps, 
                             audio_nbytes=audio_nbytes, audio_buffersize=audio_buffersize)

    def __deepcopy__(self, memo):
        return self.__copy__()

    if not TYPE_CHECKING:
        def __getattr__(self, name):
            if name in _defer_attrs and self._defer_loader and not self.reader:
                self._defer_loader()  # initialize the reader
                if name in self.__dict__:
                    return getattr(self, name)
            raise AttributeError(f"'Video' object has no attribute '{name}'")

        def __getattribute__(self, name):
            if name in _defer_attrs and self._defer_loader and not self.reader:
                self._defer_loader()  # initialize the reader
            return super().__getattribute__(name)

    def get_audio_model(self)->"Audio|None":
        '''
        Get the Audio model object for the audio clip of the video.
        This is different from the `audio` attribute, which is an `AudioFileClip` object.
        '''
        if (am:=getattr(self, '__audio_model__', None)) is None:
            if not self.reader and self._defer_loader:
                self._defer_loader()
            if not self.audio:
                am = None
            else:
                from .audio import Audio
                tmp_file: str = self.reader.filename    # type: ignore
                am = Audio.Load(tmp_file)
            setattr(self, '__audio_model__', am)
        return am

    def close(self):
        self._defer_loader = None
        if self.reader:
            self.reader.close()
            self.reader = None
        try:
            if self.audio:
                self.audio.close()
                self.audio = None
        except AttributeError:  # pragma: no cover
            pass

    def to_bytes(self, format: VideoCommonFormats|None=None)-> bytes:
        if format:
            format = _tidy_video_format(format, raise_err=False)
        
        # not yet loaded, can get from source directly
        if (not format or (format and format==self._origin_format)) and self._defer_loader:
            source: AcceptableFileSource = getattr(self._defer_loader, '__source__', None)  # type: ignore
            if source:
                if isinstance(source, bytes):
                    return source
                elif isinstance(source, Path):
                    if not os.path.exists(source):
                        raise FileNotFoundError(f'File not found: {source}')
                    with open(source, 'rb') as f:
                        return f.read()
                elif isinstance(source, BytesIO):
                    source.seek(0)
                    return source.read()
                elif isinstance(source, str):
                    if len(source) < 1024 and os.path.exists(source):
                        with open(source, 'rb') as f:
                            return f.read()
                    elif source.startswith('data:video/'):
                        comma_idx = source.find(',')
                        if comma_idx != -1:
                            b64_data = source[comma_idx+1:]
                            return base64.b64decode(b64_data)
                    elif len(source) > 2048 and len(source) % 4 == 0 and not (source.startswith(('http://', 'https://', 'ftp://', 's3://', 'gs://'))):
                        try:
                            return base64.b64decode(source)
                        except:
                            pass
        
        raw_inp_format = format
        format = self._origin_format or 'mp4'   # type: ignore
        if ((format and format == self._origin_format) or (not raw_inp_format)) and self.reader:
            temp_file = self.reader._temp_file  # type: ignore
            if not temp_file:
                with open(self.reader.filename, 'rb') as f:     # type: ignore
                    return f.read()
            else:
                with open(temp_file.name, 'rb') as f:    # type: ignore
                    return f.read()
        
        codec = _VideoTypeCodecMap.get(format, None)    # type: ignore
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp_file:
            temp_path = tmp_file.name
        try:
            self.write_videofile(
                temp_path,
                codec=codec,
                logger=None
            )
            with open(temp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def to_base64(self, format: VideoCommonFormats|None=None, url_scheme: bool=False)->str:
        if format:
            format = _tidy_video_format(format, raise_err=False)
        else:
            if url_scheme:
                if self._origin_format is not None:
                    format = self._origin_format    # type: ignore
                else:
                    format = 'mp4'
        data = self.to_bytes(format=format)
        data = base64.b64encode(data).decode('utf-8')
        if url_scheme:
            return f'data:video/{format};base64,{data}'
        return data
    
    @classmethod
    def CastVideoClip(cls, clip: VideoClip)->Self:
        if isinstance(clip, cls):
            return clip
        setattr(clip, '__class__', cls)
        clip.reader = None  # type: ignore
        clip._defer_loader = None   # type: ignore
        clip._origin_format = None  # type: ignore
        return clip  # type: ignore
        
    @classmethod
    def Load(cls, source: AcceptableFileSource|VideoClip, **kwargs)->Self:
        if isinstance(source, VideoClip):
            return cls.CastVideoClip(source)
        else:
            if not check_value_is(source, AcceptableFileSource):
                raise TypeError(f'Cannot load Video from type: {type(source)}')
            return cls(source, **kwargs)  # type: ignore
        
    @classmethod
    async def ALoad(cls, source: AcceptableFileSource|VideoClip, **kwargs)->Self:   # for compatibility
        return cls.Load(source, **kwargs)
    
    
__all__ = ['Video']


if __name__ == '__main__':
    p = r"C:\Users\MSI-NB\Downloads\成品.mp4"
    from .audio import Audio
    audio = Audio.Load(p)
    del audio
    
    with open(p, 'rb') as f:
        data = f.read()
    video = Video.Load(data)
    # video = Video.Load(p)
    
    print(video.duration, video.fps, video.size)
    print(len(video.to_bytes()))
    print(len(video.to_base64()))    
    print(len(video.to_bytes('avi')))
    print(len(video.to_base64('mov')))