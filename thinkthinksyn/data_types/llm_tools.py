import inspect

from functools import lru_cache
from pydantic import TypeAdapter, BaseModel
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar, Sequence,
                    overload, get_origin, get_args, Annotated)

if TYPE_CHECKING:
    from .completion import ToolInfo, ToolParam

@dataclass
class LLMTool:
    '''Representation of a tool that can be used by the LLM.'''

    name: str
    description: str|None = None
    params: dict[str, "ToolParam"] = field(default_factory=dict)
    return_type: "ToolParam|None" = None

    tool_creation_params: dict[str, Any]|None = None
    internal_key: str|None = None

    def __dump__(self)->"ToolInfo|str":
        if self.internal_key:
            return self.internal_key
        data = {
            'name': self.name,
            'description': self.description,
            'params': self.params,
        }
        if self.return_type:
            data['return_type'] = self.return_type
        if self.tool_creation_params:
            data['tool_creation_params'] = self.tool_creation_params
        return data     # type: ignore

# default internal tools supported in ThinkThinkSyn
_internal_common_tool_prefix = 'common'

google_search_tool = LLMTool(
    name='Google Search',
    description='Search for a topic on google.',
    internal_key=f'{_internal_common_tool_prefix}.GoogleSearch',
    params={
        'query': {'type': 'string', 'description': 'query to look up on Google', 'param_type': 'required'},
        'top_k': {'type': 'integer', 'description': 'number of results to return', 'minimum': 1, 'maximum': 10, 'default': 3, 'param_type': 'optional'},
    },
    return_type={
        'description': 'Return object of GoogleSearch tool', 
        'properties': {
            'title': {'title': 'Title', 'type': 'string'}, 
            'url': {'title': 'Url', 'type': 'string'}, 
            'snippet': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Snippet'}
        }, 
        'required': ['title', 'url'], 
        'title': 'GoogleSearchResult', 
        'type': 'object',
    }
)

wiki_search_tool = LLMTool(
    name='Wiki Search',
    description='Search for a topic on wikipedia, and return the title, summary, ... of the wiki page.',
    internal_key=f'{_internal_common_tool_prefix}.WikiSearch',
    params={
        'query': {'type': 'string', 'description': 'query to look up on wikipedia', 'param_type': 'required'},
        'lang': {'type': 'string', 'description': 'language code, usually follows ISO_639-1 standard. For cantonese, use `zh-yue`. Correct language may allows better search results.', 'default': 'en', 'param_type': 'optional'},
        'top_k': {'type': 'integer', 'description': 'number of results to return', 'minimum': 1, 'maximum': 16, 'default': 3, 'param_type': 'optional'},
    },
    return_type={
        'description': 'Return object of WikiSearch tool', 
        'properties': {
            'title': {'title': 'Title', 'type': 'string'}, 
            'summary': {'title': 'Summary', 'type': 'string'}, 
            'url': {'title': 'Url', 'type': 'string'}
        }, 
        'required': ['title', 'summary', 'url'], 
        'title': 'WikiSearchReturn', 
        'type': 'object'
    }
)

daily_weather_tool = LLMTool(
    name='Get Daily Weather',
    internal_key=f'{_internal_common_tool_prefix}.GetDailyWeather',
    description='Get a day\'s weather forecast of a city(or within n future days, n<=3). E.g. when day_count=2, returns today\'s and tomorrow\'s forecast.',
    params={
        'city': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'description': 'city or location to look up weather information, e.g. "New York". If leave it as None, it will use the current location of user. Default = None.', 'param_type': 'optional'},
        'day_count': {'type': 'integer', 'description': 'number of future days to look up weather information.', 'minimum': 1, 'maximum': 3, 'default': 1, 'param_type': 'optional'},
    },
    return_type={
        '$defs': {
            'DailyForecast': {
                'description': "Information of a day's weather forecast.", 
                'properties': {
                    'country': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}], 
                        'default': None, 
                        'title': 'Country'
                    }, 
                    'location': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}], 
                        'default': None, 
                        'title': 'Location'
                    },
                    'date': {
                        'format': 'date', 
                        'title': 'Date',
                        'type': 'string'
                    }, 
                    'sunlight': {
                        'anyOf': [{'type': 'number'}, {'type': 'null'}], 
                        'default': None, 
                        'title': 'Sunlight'
                    }, 
                    'moon_illumination': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}], 
                        'default': None, 
                        'title': 'Moon Illumination'
                    }, 
                    'temperature': {
                        'title': 'Temperature', 
                        'type': 'integer'
                    }, 
                    'lowest_temperature': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}], 
                        'default': None, 'title': 'Lowest Temperature'
                    }, 
                    'highest_temperature': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}], 
                        'default': None, 
                        'title': 'Highest Temperature'
                    }, 
                    'snowfall': {
                        'anyOf': [{'type': 'number'}, {'type': 'null'}], 
                        'default': None, 
                        'title': 'Snowfall'
                    }
                }, 
                'required': ['date', 'temperature'], 
                'title': 'DailyForecast', 
                'type': 'object'
            }
        }, 
        'items': {'$ref': '#/$defs/DailyForecast'}, 
        'type': 'array'
    }
)
# endregion

_P = ParamSpec("_P")
_R = TypeVar("_R")
_empty = inspect.Parameter.empty

@lru_cache
def _get_type_adapter(tp: Any)->TypeAdapter:
    return TypeAdapter(tp)

@lru_cache
def _get_json_schema(tp: Any)->dict[str, Any]:
    adapter = _get_type_adapter(tp)
    return adapter.json_schema()

@dataclass(kw_only=True)
class LocalLLMTool(LLMTool, Generic[_P, _R]):
    '''A local function as a tool for the LLM.'''

    func: Callable[_P, _R]
    '''the function to be used as a tool.'''

    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs)->_R:
        return self.func(*args, **kwargs)

    @overload
    @classmethod
    def Build(cls, func: Callable[_P, _R], /)->"LocalLLMTool[_P, _R]": ...

    @overload
    @classmethod
    def Build(
        cls, 
        /, 
        name: str|None = None, 
        description: str|None = None,
        hidden_params: Sequence[str]|str|None = None
    )->"Callable[[Callable[_P, _R]], LocalLLMTool[_P, _R]]": ...

    @classmethod
    def Build(  # type: ignore
        cls, 
        func = None, 
        /, 
        name: str|None = None, 
        description: str|None = None,
        hidden_params: Sequence[str]|str|None = None,
    )->"LocalLLMTool[_P, _R]":
        '''
        Build a LocalLLMTool from a function.
        Info like params, return_type will be inferred from the function's type hints.

        Args:
            name: The name of the tool. If not provided, the function's __qualname__ or __name__ will be used.
            description: The description of the tool. If not provided, the function's docstring will be used.
            hidden_params: A list of parameter names to be hidden from the tool's parameters.

        You can simply use this as a decorator:
        ```python
        @LLMTool.Build
        def add(a: int, b: int) -> int:
            return a + b
        ```

        You can gives extra param info by using `Annotated` in the function's type hints, e.g.:
        ```python
        @LLMTool.Build(name='add_numbers')
        def add(a: Annotated[int, {'description': 'The first number to add.'}], 
                b: Annotated[int, {'description': 'The second number to add.'}]) -> Annotated[int, {'description': 'The sum of the two numbers.'}]:
            """Add two numbers."""
            return a + b
        ```
        '''
        if not func:
            def wrapper(f: Callable[_P, _R])->LocalLLMTool[_P, _R]:
                return cls.Build(   # type: ignore
                    f, 
                    name=name, 
                    description=description,
                    hidden_params=hidden_params,
                )
            return wrapper  # type: ignore

        if not name:
            name = getattr(func, '__qualname__', getattr(func, '__name__', str(func)))
        if not description:
            description = inspect.getdoc(func)
        sig = inspect.signature(func)
        if isinstance(hidden_params, str):
            hidden_params = [hidden_params]
        else:
            hidden_params = hidden_params or []
        tool_params = {}
        for n, param in sig.parameters.items():
            if n in hidden_params:
                continue
            real_type = param.annotation
            real_origin = get_origin(real_type)
            extra_param_info = {}
            while real_origin is Annotated:
                param_anno_args = get_args(real_type)
                real_type = param_anno_args[0]
                for extra in param_anno_args[1:]:
                    if isinstance(extra, dict):
                        extra_param_info.update(extra)
                real_origin = get_origin(real_type)

            param_schema = _get_json_schema(real_type).copy()
            param_schema.update(extra_param_info)
            param_type = 'required'
            if ((default:=param.default) is not _empty):
                param_type = 'optional'
                if isinstance(default, BaseModel):
                    default = default.model_dump()
                param_schema['default'] = default
            param_schema['param_type'] = param_type
            tool_params[n] = param_schema  # type: ignore
        
        return_type = None
        if sig.return_annotation is not _empty:
            real_type = sig.return_annotation
            real_origin = get_origin(real_type)
            extra_return_info = {}
            while real_origin is Annotated:
                return_anno_args = get_args(real_type)
                real_type = return_anno_args[0]
                for extra in return_anno_args[1:]:
                    if isinstance(extra, dict):
                        extra_return_info.update(extra)
                real_origin = get_origin(real_type)
            return_schema = _get_json_schema(real_type).copy()
            return_schema.update(extra_return_info)
            return_schema['param_type'] = 'return'
            return_type = return_schema  # type: ignore

        return cls(
            name=name,  # type: ignore
            description=description,
            func=func,
            params=tool_params,
            return_type=return_type,    # type: ignore
        )



__all__ = [
    'LLMTool',
    'google_search_tool',
    'wiki_search_tool',
    'daily_weather_tool',

    'LocalLLMTool',
]