from typing import Literal, TypeAlias
from typing_extensions import Required
from .base import AIInput, AIOutput

OverflowHandleMode: TypeAlias = Literal['chunk', 'truncate', 'raise', 'ignore']
'''
How to handle texts that exceed the max token count.
Mode:
    - 'chunk': chunk the text into multiple parts. The final embedding will be the average of all parts.
    - 'truncate': truncate the text to the max token count.
    - 'raise': raise an error.
    - 'ignore': do nothing.
Note: this param only works when inference is calling through `Service.Serve`.
'''

class EmbeddingInput(AIInput, total=False):
    '''Input for embedding generation.'''
    
    text: Required[str]
    '''the target text'''
    simplify_zh: bool
    '''
    Change all traditional Chinese characters to simplified Chinese characters,
    This will usually increasing the performance since most models has better ability 
    to handle simplified Chinese. Cantonese will also be translated to Mandarin with 
    a simple replacement process.
    
    Default is True.
    '''
    on_overflow: OverflowHandleMode
    '''
    How to handle texts that exceed the max token count.
    Mode:
        - 'chunk': chunk the text into multiple parts. The final embedding will be the average of all parts.
        - 'truncate': truncate the text to the max token count.
        - 'raise': raise an error.
        - 'ignore': do nothing.
    Note: this param only works when inference is calling through `Service.Serve`.
    '''

class EmbeddingOutput(AIOutput[EmbeddingInput]):
    '''Output for embedding generation.'''
    
    embedding: list[float]
    '''the real embedding of the text.'''
    token_count: int
    '''Number of tokens in the text'''
    
    
__all__ = [
    "OverflowHandleMode",
    "EmbeddingInput", 
    "EmbeddingOutput", 
]