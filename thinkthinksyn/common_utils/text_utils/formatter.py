import re
import json
import json_repair

from pydantic import BaseModel
from typing import (Final, TYPE_CHECKING, overload, Literal, Generator, AsyncGenerator, AsyncIterable, 
                    Iterable, Sequence, TypeAlias)


def readable_formatting(title: str):
    '''
    Return a readable title from a snake_case string.
    e.g. `_hello_world_` -> `Hello World`
    '''
    # remove `_` in the beginning & end
    title = title.strip('_')
    if not title:
        raise ValueError('Empty title after stripping `_`')
    
    title = re.sub(r'(?<=[_-])(\w)', lambda x: x.group(1).upper(), title)
    title = title.replace('_', ' ').replace('-', ' ')
    title = title[0].upper() + title[1:]
    
    # split camel case
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
    
    return title

def to_camel_case(title: str):
    '''
    Return a camel case string from a snake_case string.
    e.g. `_hello_world_` -> `HelloWorld`
    '''
    return readable_formatting(title).replace(' ', '')

def to_snake_case(title: str):
    '''
    Return a snake case string from a camel_case string.
    e.g. `HelloWorld` -> `hello_world`
    NOTE: `.lower()` is applied to the final result.
    '''
    title = title.replace(' ', '_')
    title = re.sub(r'(?<=[a-z])([A-Z])', r'_\1', title)
    title = title.lower()
    return title

INVALID_NONE_STRS: Final[tuple[str, ...]] = ('none', 'none]', 'none[')

_RECURSIVE_TYPES = (dict, list)

def _recursive_repair(data):
    if isinstance(data, str):
        if data.lower() in INVALID_NONE_STRS:
            return None
    elif isinstance(data, dict):
        for k in tuple(data.keys()):
            v = data[k]
            if isinstance(v, str):
                if v.lower() in INVALID_NONE_STRS:
                    data[k] = None
            elif isinstance(v, _RECURSIVE_TYPES):
                data[k] = _recursive_repair(v)
    elif isinstance(data, list):
        for i in range(len(data)):
            v = data[i]
            if isinstance(v, str):
                if v.lower() in INVALID_NONE_STRS:
                    data[i] = None
            elif isinstance(v, _RECURSIVE_TYPES):
                data[i] = _recursive_repair(v)
    return data
        
def json_repair_loads(s: str):
    '''
    Repair common json errors in the string and return the final json object.
    e.g.
        `None` -> `null`
        `'` -> `"`
    '''
    if TYPE_CHECKING:
        return json.loads(s)    # for type hinting only
    data = json_repair.loads(s)
    return _recursive_repair(data)

def extract_json_dict(s: str)->dict|None:
    '''Try to extract & fix a json dict from the string, return None if failed.
    This function is useful for extracting json return from LLM.'''
    json_pattern = r'\{.*\}'
    if (m:=re.search(json_pattern, s, re.DOTALL)):
        try:
            return json_repair_loads(m.group())
        except:
            return None

def full_width_text_tidy_up(text:str):
    '''
    Tidy full-width text to normal text, e.g. '１２３４５６７８９０' -> '1234567890' 
    '''
    small_case = {u'\uff41': 'a', u'\uff42': 'b', u'\uff43': 'c', u'\uff44': 'd',
                  u'\uff45': 'e', u'\uff46': 'f', u'\uff47': 'g', u'\uff48': 'h',
                    u'\uff49': 'i', u'\uff47': 'j', u'\uff4b': 'k', u'\uff4c': 'l',
                    u'\uff4d': 'm', u'\uff4e': 'n', u'\uff4f': 'o', u'\uff50': 'p',
                    u'\uff51': 'q', u'\uff52': 'r', u'\uff53': 's', u'\uff54': 't',
                    u'\uff55': 'u', u'\uff56': 'v', u'\uff57': 'w', u'\uff58': 'x',
                    u'\uff59': 'y', u'\uff5a': 'z'}
    cap_case = {u'\uff21': 'A', u'\uff22': 'B', u'\uff23': 'C', u'\uff24': 'D',
                u'\uff25': 'E', u'\uff26': 'F', u'\uff27': 'G', u'\uff28': 'H',
                u'\uff29': 'I', u'\uff2a': 'J', u'\uff2b': 'K', u'\uff2c': 'L',
                u'\uff2d': 'M', u'\uff2e': 'N', u'\uff2f': 'O', u'\uff30': 'P',
                u'\uff31': 'K', u'\uff32': 'R', u'\uff33': 'S', u'\uff35': 'T',
                u'\uff35': 'U', u'\uff36': 'V', u'\uff37': 'W', u'\uff38': 'X',
                u'\uff39': 'Y', u'\uff3a': 'Z'}
    num = {u'\uff11': '1', u'\uff12': '2', u'\uff13': '3', u'\uff14': '4',
            u'\uff15': '5', u'\uff16': '6', u'\uff17': '7', u'\uff18': '8',
            u'\uff19': '9', u'\uff10': '0'}
    symbols = {u'\u3000': ' ', 
               u'\uff01': '!', u'\uff02': '"', u'\uff03': '#', u'\uff04': '$',
               '\uff05': '%', u'\uff06': '&', u'\uff07': "'", u'\uff08': '(',
                u'\uff09': ')', u'\uff0a': '*', u'\uff0b': '+', u'\uff0c': ',',
                u'\uff0d': '-', u'\uff0e': '.', u'\uff0f': '/', u'\uff1a': ':',
                u'\uff1b': ';', u'\uff1c': '<', u'\uff1d': '=', u'\uff1e': '>',
                u'\uff1f': '?', u'\uff20': '@', u'\uff3b': '[', u'\uff3c': '\\',
                u'\uff3d': ']', u'\uff3e': '^', u'\uff3f': '_', u'\uff40': '`',
                u'\uff5b': '{', u'\uff5c': '|', u'\uff5d': '}', u'\uff5e': '~'}
    
    for k, v in small_case.items():
        text = text.replace(k, v)
    for k, v in cap_case.items():
        text = text.replace(k, v)
    for k, v in num.items():
        text = text.replace(k, v)
    for k, v in symbols.items():
        text = text.replace(k, v)
    return text

def tidy_unbalanced_brackets(s: str):
    '''Tidy unbalanced brackets in the string, e.g. 'a(b' -> 'ab' '''
    pairs = (('(', ')'), ('[', ']'), ('{', '}'), ('<', '>'))
    for l, r in pairs:
        l_indices = []
        r_indices = []
        for i, c in enumerate(s):
            if c == l:
                l_indices.append(i - len(l_indices))
            elif c == r:
                r_indices.append(i - len(r_indices))
        r_indices = [i for i in reversed(r_indices)]
        if len(l_indices) != len(r_indices):
            if len(l_indices) > len(r_indices):
                for i in l_indices[:len(l_indices) - len(r_indices)]:
                    s = s[:i] + s[i + 1:]
            else:
                for i in r_indices[:len(r_indices) - len(l_indices)]:
                    s = s[:i] + s[i + 1:]
    return s

@overload
def get_num_from_text(s: str) -> int|float|None:...
@overload
def get_num_from_text(s: str, raise_error: Literal[False]=False) -> int|float|None:...
@overload
def get_num_from_text(s: str, raise_error: Literal[True]=True) -> int|float:...

def get_num_from_text(s: str, raise_error: bool = False):
    '''
    Convert a string to a number. Supported formats:
    - Hexadecimal: '0x10' -> 16
    - Binary: '0b1010' -> 10
    - Octal: '0o12' -> 10
    - Decimal: '123.456' -> 123.456
    - Chinese numbers: '四百二十' -> 420
    '''
    if s.startswith('0x'):
        try:
            return int(s, 16)
        except:
            ...
    elif s.startswith('0b'):
        try:
            return int(s, 2)
        except:
            ...
    elif s.startswith('0o'):
        try:
            return int(s, 8)
        except:
            ...
    if s.isnumeric():
        from .zh_num_convertor import chinese2number_within_text
        try:
            return int(chinese2number_within_text(s))
        except:
            ...
    try:
        return float(s)
    except:
        ...
    if raise_error:
        raise ValueError(f'Cannot convert `{s}` to number')
    return None

# region streaming html tag extractor
class StreamTagExtraction(BaseModel):
    '''Dataclass for extraction from` streaming_tag_extractor` and `async_streaming_tag_extractor` methods.'''
    
    data: str = ''
    '''The content of the tag, or the text outside of tags.'''
    tag: str|None = None
    '''The tag name, or None if the content is not within a tag.'''
    event: Literal['tag_start', 'tag_content', 'tag_end', 'none'] = 'none'
    '''The event type, can be one of:
        - 'tag_start': the start of a tag, e.g. `<div>`
        - 'tag_content': the content within a tag, e.g. `Hello World`
        - 'tag_end': the end of a tag, e.g. `</div>`
        - 'none': the content is not within a tag, e.g. `Hello World
    '''
    
    def __getitem__(self, item: str|int):
        '''Get the value of the attribute by its name.'''
        if item in ('data', 0):
            return self.data
        elif item in ('tag', 1):
            return self.tag
        elif item in ('event', 2):
            return self.event
        else:
            raise KeyError(f'Invalid key: {item}')
        
    def __sse_event__(self):
        '''for `sse_response` method in `utils.server.response` module.'''
        if not self.tag:
            return ('message', self.data)
        return (self.tag, self.data)

class StreamTagConfig(BaseModel):
    '''
    Configuration for a tag to be extracted by `streaming_tag_extractor` and `async_streaming_tag_extractor` methods.
    NOTE: it is not necessary to provide a HTML tag. start/end/tag_name can be any string.
    '''
    
    start: str
    '''Start pattern of the tag, e.g. `<div>`.'''
    end: str = ''
    '''end pattern of the tag, e.g. `</div>`. If not provided, it will equal to `start` by default.'''
    tag_name: str = ''
    '''The name of the tag, e.g. `div`. If not provided, it will be extracted from `start` tag.'''
    
    def __eq__(self, value: object) -> bool:
        return super().__eq__(value)
    
    def __hash__(self) -> int:
        return hash((self.start, self.end))
    
    def model_post_init(self, context) -> None:
        if not self.start:
            raise ValueError('Start tag cannot be empty')
        if not self.end:
            if self.start.startswith('<') and self.start.endswith('>'):
                self.end = f'</{self.start[1:-1]}>'
                if not self.tag_name:
                    self.tag_name = self.start[1:-1]
            else:
                self.end = self.start
        if not self.tag_name:
            self.tag_name = self.start
            
_TextGeneratorT: TypeAlias = Generator[str, None, None]| Iterable[str]
_AsyncTextGeneratorT: TypeAlias = AsyncGenerator[str, None]| AsyncIterable[str]
_SyncOrAsyncTextGeneratorT: TypeAlias = _TextGeneratorT | _AsyncTextGeneratorT
_MAX_UNKNOWN_HTML_TAG_LENGTH: Final[int] = 64
_TagData: TypeAlias = tuple[str, str, str] # (start, end, tag_name)

def _extract_tag(chunk: str, valid_tags: set[StreamTagConfig], context: dict)->Generator[StreamTagExtraction, None, None]:
    chars = chunk
    
    is_first_call = context.get('is_first_call', True)
    is_last_call = context.get('is_last_call', False)
    yield_prefix = context.get('yield_prefix', True)
    buf = context.get('buf', '')
    
    if is_first_call:
        context['is_first_call'] = False
        prefix = context.get('prefix', '')
        chars = prefix + chars
        context['tag_stack'] = []
        context['not_end_tag'] = False
        if valid_tags:
            context['options'] = [] # possible tags
            context['min_start_len'] = min(len(t.start) for t in valid_tags)    # minimum length of start tag
            possible_ends: dict[int, set[str]] = {} # {position: set of possible chars}
            for i in range(max(len(t.end) for t in valid_tags)):
                possible_ends[i] = set([t.end[i] for t in valid_tags if len(t.end) > i])
            context['possible_ends'] = possible_ends
            
            possible_starts: dict[tuple[int, str], set[StreamTagConfig]] = {}   # {(position, char): set of tags that can start with this char at this position}
            for t in valid_tags:
                for i, c in enumerate(t.start):
                    if (i, c) not in possible_starts:
                        possible_starts[(i, c)] = set()
                    possible_starts[(i, c)].add(t)
            context['possible_starts'] = possible_starts
           
    tag_stack: list[_TagData] = context['tag_stack']
    curr_tag = None
    if tag_stack:
        curr_tag = tag_stack[-1][2]
    
    should_yield = not (is_first_call and not yield_prefix)
    
    while len(chars):
        c, chars = chars[0], chars[1:]  # get the first character and the rest of the string
        buf += c
        state: Literal['tag_start', 'tag_end', None] = context.get('state', None)
        if not valid_tags:  # extracting all html tags
            if state == 'tag_start':
                if c == '>':
                    if buf[-1] == '/':
                        if (len(buf)>2): # no content tag, e.g. `<br/>`
                            tn = buf[1:-2]
                            if should_yield:
                                yield StreamTagExtraction(data='', tag=tn, event='tag_start')
                                yield StreamTagExtraction(data='', tag=tn, event='tag_end')
                        else:
                            if should_yield:
                                yield StreamTagExtraction(data=buf, tag=curr_tag, event=('tag_content' if curr_tag else 'none'))
                    else:
                        tn = buf[1:-1]
                        if should_yield:
                            yield StreamTagExtraction(data='', tag=tn, event='tag_start')
                        tag_stack.append((f'<{tn}>', f'</{tn}>', tn))
                        curr_tag = tn
                    context['state'] = None
                    buf = ''
                elif c == '/' and len(buf)>2: # e.g. `<div/>`
                    continue
                else:
                    tn = buf[1:]
                    if (not tn.isidentifier()) or (len(tn) > _MAX_UNKNOWN_HTML_TAG_LENGTH):
                        if should_yield:
                            yield StreamTagExtraction(data=buf, tag=curr_tag, event=('tag_content' if curr_tag else 'none'))
                        buf = ''
                        context['state'] = None
                    else:
                        continue
            elif state == 'tag_end':
                if c == '>':
                    tn = buf[2:-1]
                    if tn != curr_tag:
                        if should_yield:
                            yield StreamTagExtraction(data=buf, tag=curr_tag, event=('tag_content' if curr_tag else 'none'))
                        buf = ''
                    else:
                        tag_stack.pop()
                        if should_yield:
                            yield StreamTagExtraction(data='', tag=curr_tag, event='tag_end')
                        buf = ''
                        if tag_stack:
                            curr_tag = tag_stack[-1][2]
                        else:
                            curr_tag = None
                    context['state'] = None
                elif c == '/':
                    if buf[-2] == '<':
                        continue
                    else:
                        if should_yield:
                            yield StreamTagExtraction(data=buf, tag=curr_tag, event=('tag_content' if curr_tag else 'none'))
                        buf = ''
                        context['state'] = None
                else:
                    partial_tn = buf[2:]
                    if (len(buf) == 2) or (not ((curr_tag or '').startswith(partial_tn))):
                        context['not_end_tag'] = True
                        chars = buf + chars  # put characters back
                        buf = ''
                        context['state'] = None
                    else:
                        continue
            else:
                if c != '<':
                    if should_yield:
                        yield StreamTagExtraction(data=buf, tag=curr_tag, event=('tag_content' if curr_tag else 'none'))
                    buf = ''
                    context['not_end_tag'] = False
                else:
                    if tag_stack and (not context['not_end_tag']): # having unclosed tags
                        context['state'] = 'tag_end'
                    else:
                        context['state'] = 'tag_start'
                        context['not_end_tag'] = False
                        
        else: # only extract valid tags
            possible_starts = context['possible_starts']
            if state == 'tag_start':
                new_c = possible_starts.get((len(buf)-1, c), None)
                old_c = context['candidates']
                options: list[StreamTagConfig] = context['options']
                if new_c and old_c and (remain_c := old_c.intersection(new_c)):
                    for t in remain_c:
                        if t.start == buf:
                            options.append(t)
                    context['candidates'] = remain_c
                else:
                    # got invalid character
                    context['state'] = None
                    del context['candidates']
                    if options:
                        selected = sorted(options, key=lambda x: len(x.start), reverse=True)[0]
                        if should_yield:
                            yield StreamTagExtraction(data='', tag=selected.tag_name, event='tag_start')
                        tag_stack.append((selected.start, selected.end, selected.tag_name))
                        curr_tag = selected.tag_name
                        chars = buf[len(selected.start):] + chars  # put remaining characters back
                        options.clear()
                    else:
                        if should_yield:
                            yield StreamTagExtraction(data=buf[0], tag=curr_tag, event=('tag_content' if curr_tag else 'none'))
                        chars = buf[1:] + chars  # put characters back
                    buf = ''            
            elif state == 'tag_end':
                end_chars = context['possible_ends'].get(len(buf)-1, None)
                if (not tag_stack) or (not end_chars) or (c not in end_chars) or (not tag_stack[-1][1].startswith(buf)):
                    # invalid char for tag end
                    context['state'] = None
                    chars = buf + chars  # put characters back
                    buf = ''
                    context['not_end_tag'] = True
                elif tag_stack[-1][1] == buf:  # valid end
                    tn = tag_stack[-1][2]
                    if should_yield:
                        yield StreamTagExtraction(data='', tag=tn, event='tag_end')
                    tag_stack.pop()
                    if tag_stack:
                        curr_tag = tag_stack[-1][2]
                    else:
                        curr_tag = None
                    context['state'] = None
                    buf = ''
                else:
                    continue    # continue to collect characters for the tag end
            else:
                if not context['not_end_tag'] and tag_stack and tag_stack[-1][1].startswith(c):
                    context['state'] = 'tag_end'
                elif (0, c) in possible_starts:
                    context['not_end_tag'] = False
                    context['state'] = 'tag_start'
                    context['candidates'] = possible_starts[(0, c)].copy()
                    if context['min_start_len'] == 1:
                        for t in context['candidates']:
                            if t.start == c:
                                context['options'].append(t)
                else:
                    context['not_end_tag'] = False
                    if should_yield:
                        yield StreamTagExtraction(data=buf, tag=curr_tag, event=('tag_content' if curr_tag else 'none'))
                    buf = ''

    if is_last_call:
        if valid_tags and (curr_options := context.get('options', None)):
            selected = sorted(curr_options, key=lambda x: len(x.start), reverse=True)[0]
            yield StreamTagExtraction(data='', tag=selected.tag_name, event='tag_start')
        elif buf:
            tag_stack = context.get('tag_stack', [])
            if tag_stack:
                last_tag = tag_stack[-1][2]
            else:
                last_tag = None
            yield StreamTagExtraction(data=buf, tag=last_tag, event=('tag_content' if last_tag else 'none'))
    
    context['buf'] = buf
    
def _tidy_tags(tags: Sequence[str|tuple[str, str]|tuple[str,str,str]|StreamTagConfig]|None) -> set[StreamTagConfig]:
    if tags is None:
        return set()
    if isinstance(tags, (str, StreamTagConfig)):
        tags = [tags]
    tidied = []
    added_tag_start = {}  # to avoid duplicate start tags
    for t in tags:
        if isinstance(t, str):
            key = t.strip()
            if not key:
                raise ValueError('Got empty string for XML tag')
            if key.startswith('<') and key.endswith('>'):
                key = key[1:-1].strip('/').strip()
            if not key.isidentifier():
                raise ValueError(f'Invalid tag: {t}')
            tag = StreamTagConfig(start=f'<{key}>', end=f'</{key}>', tag_name=key)
            if tag.start not in added_tag_start:
                added_tag_start[tag.start] = hash(tag)
                tidied.append(tag)
            else:
                if hash(tag) != added_tag_start[tag.start]:
                    raise ValueError(f'Duplicate tag start pattern: {tag.start}')
        elif isinstance(t, StreamTagConfig):
            if t.start in added_tag_start:
                if hash(t) != added_tag_start[t.start]:
                    raise ValueError(f'Duplicate tag start pattern: {t.start}')
            else:
                added_tag_start[t.start] = hash(t)
                tidied.append(t)
        elif isinstance(t, (tuple, list)):
            assert len(t) in (2,3), 'Tuple must have exactly 2 or 3 elements, which are (start, end, tag_name(optional))'
            assert all(isinstance(x, str) for x in t), 'Tuple elements must be strings'
            assert t[0], 'Tag start pattern cannot be empty'
            if len(t) == 2:
                tag = StreamTagConfig(start=t[0], end=t[1])
            else:
                tag = StreamTagConfig(start=t[0], end=t[1], tag_name=t[2])
            if tag.start in added_tag_start:
                if hash(tag) != added_tag_start[tag.start]:
                    raise ValueError(f'Duplicate tag start pattern: {tag.start}')
            else:
                added_tag_start[tag.start] = hash(tag)
                tidied.append(tag)
        else:
            raise TypeError(f'Invalid tag type: {type(t)}')
    return set(tidied)

def streaming_tag_extractor(
    generator: _TextGeneratorT,
    valid_tags: Sequence[str|StreamTagConfig|tuple[str, str]|tuple[str, str, str]]|None=None,
    prefix: str='',
    yield_prefix: bool = True,
)->Generator[StreamTagExtraction, None, None]:
    '''
    Extract contents & tags from a text generator (async version is `async_streaming_tag_extractor`), 
    e.g. 
    ```
    stream: ['<a', '>H', 'ello', '</a>', 'World']
        ---> 
            output: (data='', tag='a', event='tag_start'), (data='Hello', tag='a', event='tag_content'), 
                                (data='', tag='a', event='tag_end'), (data='World', tag=None, event='none')
    ```
    
    Args:
        generator: A synchronous or asynchronous generator that yields text chunks.
        valid_tags: A sequence of valid tags to extract. It can be:
            - (str) xml tag, e.g. `div`(`<` and `>` will be added for you automatically if not appear).
            - A `StreamTagConfig` object to define special start/end pattern and custom tag name.
            - A tuple of two strings representing a start and end tag, e.g. `('<div>', '</div>')`.
            - A tuple of three strings representing a start tag, end tag, and tag name, e.g. `('<div>', '</div>', 'div')`.
        prefix: A prefix to add before first chunk of text, e.g. `<div>`. This helps to trigger
              next `</div>` even start tag `<div>` is not provided by the generator. 
        yield_prefix: if prefix is provided, whether to yield events within the prefix. e.g. if you provided
                    `<div>` as prefix and `yield_prefix=True`, `(data='', tag='div', event='tag_start')` will be yielded 
                    before first chunk arrived from text generator.
    '''
    tags = _tidy_tags(valid_tags)
    context = {'prefix': prefix, 'is_first_call': True, 'yield_prefix': yield_prefix}
    if prefix:
        for e in _extract_tag('', tags, context):
            yield e
    for chunk in generator:
        for e in _extract_tag(chunk, tags, context):
            yield e
    context['is_last_call'] = True
    for e in _extract_tag('', tags, context):
        yield e

async def async_streaming_tag_extractor(
    generator: _SyncOrAsyncTextGeneratorT,
    valid_tags: Sequence[str|StreamTagConfig|tuple[str, str]|tuple[str, str, str]]|None=None,
    prefix: str='',
    yield_prefix: bool = True,
)->AsyncGenerator[StreamTagExtraction, None]:
    '''
    Extract contents & tags from a text generator (async version of `streaming_tag_extractor`), 
    e.g. 
    ```
    stream: ['<a', '>H', 'ello', '</a>', 'World']
        ---> 
            output: (data='', tag='a', event='tag_start'), (data='Hello', tag='a', event='tag_content'), 
                                (data='', tag='a', event='tag_end'), (data='World', tag=None, event='none')
    ```
    
    Args:
        generator: A synchronous or asynchronous generator that yields text chunks.
        valid_tags: A sequence of valid tags to extract. It can be:
            - (str) xml tag, e.g. `div`(`<` and `>` will be added for you automatically if not appear).
            - A `StreamTagConfig` object to define special start/end pattern and custom tag name.
            - A tuple of two strings representing a start and end tag, e.g. `('<div>', '</div>')`.
            - A tuple of three strings representing a start tag, end tag, and tag name, e.g. `('<div>', '</div>', 'div')`.
        prefix: A prefix to add before first chunk of text, e.g. `<div>`. This helps to trigger
              next `</div>` even start tag `<div>` is not provided by the generator. 
        yield_prefix: if prefix is provided, whether to yield events within the prefix. e.g. if you provided
                    `<div>` as prefix and `yield_prefix=True`, `(data='', tag='div', event='tag_start')` will be yielded 
                    before first chunk arrived from text generator.
    '''
    from ..concurrent_utils import get_async_generator
    async_generator = get_async_generator(generator) # type: ignore
    tags = _tidy_tags(valid_tags)
    context = {'prefix': prefix, 'is_first_call': True, 'yield_prefix': yield_prefix}
    if prefix:
        for e in _extract_tag('', tags, context):
            yield e
    async for chunk in async_generator:
        for e in _extract_tag(chunk, tags, context):
            yield e
    context['is_last_call'] = True
    for e in _extract_tag('', tags, context):
        yield e

# endregion


__all__ = [
    'readable_formatting', 'to_camel_case', 'to_snake_case', 
    'json_repair_loads', 'extract_json_dict',
    'tidy_unbalanced_brackets', 'full_width_text_tidy_up',
    'get_num_from_text',
    'StreamTagConfig', 'StreamTagExtraction',
    'streaming_tag_extractor', 'async_streaming_tag_extractor',
]


if __name__ == '__main__':
    def test_formatting():
        print(readable_formatting('_hello_world_'))  # Hello World
        print(readable_formatting('helloWorld'))

    def test_json_repair():    
        print(json_repair_loads('{"x": None}'))
        print(json_repair_loads('[null, None]'))
        print(json_repair_loads('[{"name": "Get Daily Wether", "params": {"city": None, "day_count": 1}}]'))
        print(json_repair_loads('none'))
        print(json_repair_loads('null'))
    
    def test_extract_num():
        print(get_num_from_text('0x10'))
        print(get_num_from_text('四112'))
        print(get_num_from_text('123.321'))
        
    def test_streaming_tag_extractor():
        stream = ['<di', 'v', '>Hello', '<p>', 'World', '</p>', '!', '<', '/div', '>', '</header>']
        print('#' * 20)
        for e in streaming_tag_extractor(stream, yield_prefix=True):
            print(e)    # only `div` & `p` tags will be extracted
        print('#' * 20)
        for e in streaming_tag_extractor(stream, valid_tags=['div'], yield_prefix=True):
            print(e)    # only `div` tag will be extracted
        print('#' * 20)
        for e in streaming_tag_extractor(stream, valid_tags=['div','p'], yield_prefix=True):
            print(e)    # only `div` & `p` tags will be extracted
        print('#' * 20)
        for e in streaming_tag_extractor(stream, prefix='<header>', yield_prefix=True):
            print(e)    # div/p/header tags will all be extracted
        print('#' * 20)
        
    # test_formatting()
    # test_json_repair()
    # test_extract_num()
    # test_streaming_tag_extractor()