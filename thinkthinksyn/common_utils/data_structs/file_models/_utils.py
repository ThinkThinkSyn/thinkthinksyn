import os
import hashlib

from pydantic import BaseModel
from functools import cache

def _hash_md5(data: bytes)->str:
    m = hashlib.md5()
    m.update(data)
    return m.hexdigest()

def _try_get_from_dict(d: dict, *keys: str):
    for k in keys:
        try:
            return d[k]
        except KeyError:
            continue
    return None

def _dump_media_dict(base64_data: str, cls: type)->dict:
    return {
        'type': cls.__name__.lower(),
        'data': base64_data,
    }

class _MediaModel(BaseModel):
    type: str
    '''Indicates the media type, e.g., image, video, audio'''
    data: str
    '''base64 encoded data'''

_default_media_json_schema = _MediaModel.model_json_schema()

@cache
def _get_media_json_schema(cls: type)->dict:
    schema = _default_media_json_schema.copy()
    schema['title'] = cls.__name__
    return schema

@cache
def _init_ffmpeg():
    import ffmpeg_downloader as ffdl
    ffmpeg_path = ffdl.ffmpeg_path
    ffprobe_path = ffdl.ffprobe_path
    ffplay_path = ffdl.ffplay_path
    if not (ffmpeg_path and ffprobe_path and ffplay_path):
        from ffmpeg_downloader import __main__ as ffdl_main
        from argparse import Namespace
        args = Namespace(
            add_path=False,
            force=False,
            version=None,
            proxy=None,
            retries=3,
            timeout=30,
            no_cache_dir=False,
            y=True,
            dst=None,
            set_env=['name=ffmpeg', 'name=ffprobe', 'name=ffplay'],
            no_simlinks=False,
            presets=None,
            upgrade=False,
            reset_env=False,
        )
        ffdl_main.install(args)
        ffmpeg_path = ffdl.ffmpeg_path
        ffprobe_path = ffdl.ffprobe_path
        ffplay_path = ffdl.ffplay_path
        
    ffmpeg_dir = ffdl.ffmpeg_dir
    os.environ['FFMPEG_BINARY'] = ffmpeg_path
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    
    # override moviepy
    import moviepy.config as mpy_config
    mpy_config.FFMPEG_BINARY = ffmpeg_path
    
    # override pydub
    from pydub.utils import which as pydub_origin_which
    def pydub_which(cmd):
        if cmd == 'ffmpeg':
            return ffmpeg_path
        return pydub_origin_which(cmd)
    
    def pydub_get_encoder_name():
        return ffmpeg_path
    
    def pydub_get_player_name():
        return ffplay_path
    
    def pydub_get_prober_name():
        return ffprobe_path
    
    import pydub.utils
    pydub.utils.which = pydub_which
    pydub.utils.get_encoder_name = pydub_get_encoder_name
    pydub.utils.get_player_name = pydub_get_player_name
    pydub.utils.get_prober_name = pydub_get_prober_name