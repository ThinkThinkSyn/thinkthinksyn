from typing import Self
from pathlib import Path

class PDFPage:
    def to_base64(self, )->str:
        ...
        
class PDF:
    @property
    def pages(self)->tuple[PDFPage, ...]:
        ...
    
    @classmethod
    def Load(cls, source: str|Path|Self)->Self:
        if isinstance(source, cls):
            return cls
        ...
        
__all__ = ['PDFPage', 'PDF']