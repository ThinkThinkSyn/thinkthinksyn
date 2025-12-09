from pathlib import Path
from typing import Self, TypeAlias, Literal

VideoFormats: TypeAlias = Literal["mp4", "gif"]

async def _load_video():
    ...

class Video:
    def to_base64(self, url_scheme: bool=False)->str:
        ...
        
    @classmethod
    def Load(cls, source: str|Path|Self)->Self:
        if isinstance(source, cls):
            return source
        ...
    
    
__all__ = ['VideoFormats', 'Video']

if __name__ == '__main__':
    p = r'C:\Users\yashi\Desktop\thinkthinksyn\tmp\test2.mp4'