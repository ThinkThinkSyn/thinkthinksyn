from typing import TypedDict, Any, TypeAlias, Literal, Union, TYPE_CHECKING, Generic, TypeVar

from ..common_utils.data_structs.condition import BaseCondition, Condition

class NodeFilterPolicy(TypedDict, total=False):
    '''
    Node filter policy is a dataclass for containing configs for selecting a 
    suitable node to handle a task.
    '''
    condition: str|BaseCondition|Condition|None
    '''Condition for filtering nodes, e.g. `name==...`'''
    ignore_tier: bool
    '''If True, even `fallback` nodes can be selected to handle this task in normal cases.
    Default is False.'''

class _AIInputBase(TypedDict): ...

class _AIInput(_AIInputBase, total=False):
    model_filter: str|BaseCondition|Condition|None
    '''
    Filter for selecting a suitable model to do inference.
    e.g. `name == Qwen/Qwen3-Omni-30B-A3B-Instruct`.
    All supported operators:
        - `==`/`=`: equals
        - `!=`: not equals
        - `>`: greater than
        - `>=`: greater than or equals
        - `<`: less than
        - `<=`: less than or equals
        - `in`: in list
        - `contains`: contains substring
    
    Multiple conditions can be combined with `and`/`&&`/`||`/`or` operators.
    `not` operator is also supported.
    '''
    node_filter: str|BaseCondition|Condition|NodeFilterPolicy|None
    '''Filter for selecting a suitable node to do inference.'''
    cache: bool
    '''whether to use cached result(if any). Default is True.'''
    save_cache: bool | None
    '''
    Whether to save the cache after inference.
    If `None`, it will follow the `cache` field.
    '''
    local_cache: bool
    '''whether to use local cache(if any). Default is True.'''
    save_local_cache: bool
    '''whether to save the local cache after inference. Default is False'''
    extra_params: dict[str, Any]
    '''
    Extra parameters for the AI service(if any).
    Any unknown parameters passed in the body will be packed into `extra_params` automatically.
    '''
    
if TYPE_CHECKING:
    class AIInput(_AIInput, extra_items=Any): ...
else:
    AIInput = _AIInput

_T = TypeVar('_T')
_ContainT = TypeVar('_ContainT')
_IT = TypeVar('_IT', bound=_AIInputBase)

class AIOutput(Generic[_IT], TypedDict, total=False):
    input: _IT
    '''receipt of the final input.'''

class ConditionProxy(Generic[_T, _ContainT]):
    def __init__(self, name: str):
        self.name = name
        
    def __eq__(self, other: _T|BaseCondition) -> BaseCondition:
        if isinstance(other, BaseCondition):
            other = other.value
        return BaseCondition(self.name, '=', other)

    def __ne__(self, other: _T|BaseCondition) -> BaseCondition:
        if isinstance(other, BaseCondition):
            other = other.value
        return BaseCondition(self.name, '!=', other)
    
    def __lt__(self, other: _T|BaseCondition) -> BaseCondition:
        if isinstance(other, BaseCondition):
            other = other.value
        return BaseCondition(self.name, '<', other)
    
    def __le__(self, other: _T|BaseCondition) -> BaseCondition:
        if isinstance(other, BaseCondition):
            other = other.value
        return BaseCondition(self.name, '<=', other)
    
    def __gt__(self, other: _T|BaseCondition) -> BaseCondition:
        if isinstance(other, BaseCondition):
            other = other.value
        return BaseCondition(self.name, '>', other)
    
    def __ge__(self, other: _T|BaseCondition) -> BaseCondition:
        if isinstance(other, BaseCondition):
            other = other.value
        return BaseCondition(self.name, '>=', other)
    
    def __contains__(self, item: _ContainT|BaseCondition) -> BaseCondition:
        if isinstance(item, BaseCondition):
            item = item.value
        return BaseCondition(self.name, 'in', item)

JsonValType: TypeAlias = Literal['string', 'number', 'integer', 'boolean', 'array', 'object', 'function', 'null']
'''Type of the value in json schema'''

class _JsonSchema(TypedDict, total=False):
    '''
    Type hints for JSON Schema.
    See: https://json-schema.org/understanding-json-schema
    
    Note: indicate empty field by using `EmptyJsonField` instead of `None`.
    '''
    
    title: str
    '''name of the parameter.'''
    description: str
    '''description of the parameter'''
    default: Any
    '''the default value of the parameter'''
    examples: list[Any]
    '''examples of the parameter'''
    
    type: JsonValType|list[JsonValType]
    '''type of the value of the parameter. If it is a list, it means the value can be multiple types.'''
    multipleOf: int|float
    '''The value of the parameter should be multiple of this value
    This field is just for number type'''
    minimum: int|float
    '''The minimum value of the parameter. This field is just for number type'''
    maximum: int|float
    '''The maximum value of the parameter. This field is just for number type'''
    exclusiveMinimum: bool
    '''Whether the minimum value is exclusive. This field is just for number type. 
    If this field is given with number, it will set to be True and the value will be set to `minimum` field.'''
    exclusiveMaximum: bool
    '''Whether the maximum value is exclusive. This field is just for number type.
    If this field is given with number, it will set to be True and the value will be set to `maximum` field.'''
    enum: list[Any]
    '''The value of the parameter should be one of the values in this list.
    Note: when enum is set, val_type should be ignored, or set to the types within the enum values.'''
    uniqueItems: bool
    '''Whether the items of the array should be unique. If this field is given with array, it will set to be True.'''
    minItems: int
    '''The minimum number of items in the array. This field is just for array type'''
    items: 'JsonSchema'
    '''items of the array. This field is just for array type'''
    anyOf: list['JsonSchema']
    '''this means this schema can be any of the sub-schema within this field.'''
    allOf: list['JsonSchema']
    '''this means this schema must be all of the sub-schema within this field.'''
    properties: dict[str, 'JsonSchema']
    '''The properties of the object. The key is the name of the property, and the value is the schema of the property.'''
    required: list[str]
    '''The required properties of the object. This field is just for object type.'''
    patternProperties: dict[str, 'JsonSchema']
    '''The properties of the object. The key is the pattern of the property name, and the value is the schema of the property.
    e.g. {"^S_": {"type": "string"}}'''
    additionProperties: Union['JsonSchema', bool]
    '''whether the object can have additional properties. If it is a schema, it will be the schema of the additional properties.
    This field can be boolean or schema.'''

if TYPE_CHECKING:
    class JsonSchema(_JsonSchema, extra_items=Any): ...
else:
    JsonSchema = _JsonSchema

def tidy_json_schema(schema: dict)->JsonSchema:
    if 'name' in schema and 'title' not in schema:
        schema['title'] = schema.pop('name')
    if "val_type" in schema and "type" not in schema:
        schema['type'] = schema.pop('val_type')
    if 'multiple_of' in schema and 'multipleOf' not in schema:
        schema['multipleOf'] = schema.pop('multiple_of')
    if 'exclusive_minimum' in schema and 'exclusiveMinimum' not in schema:
        schema['exclusiveMinimum'] = schema.pop('exclusive_minimum')
    if 'exclusive_maximum' in schema and 'exclusiveMaximum' not in schema:
        schema['exclusiveMaximum'] = schema.pop('exclusive_maximum')
    if 'unique_items' in schema and 'uniqueItems' not in schema:
        schema['uniqueItems'] = schema.pop('unique_items')
    if 'min_items' in schema and 'minItems' not in schema:
        schema['minItems'] = schema.pop('min_items')
    if 'pattern_properties' in schema and 'patternProperties' not in schema:
        schema['patternProperties'] = schema.pop('pattern_properties')
    if 'additional_properties' in schema and 'additionalProperties' not in schema:
        schema['additionalProperties'] = schema.pop('additional_properties')
    if 'any_of' in schema and 'anyOf' not in schema:
        schema['anyOf'] = schema.pop('any_of')
    if 'all_of' in schema and 'allOf' not in schema:
        schema['allOf'] = schema.pop('all_of')
    
    return schema  # type: ignore



__all__ = [
    'AIInput', 
    'AIOutput',
    'NodeFilterPolicy',
    'ConditionProxy',
    'JsonValType',
    'JsonSchema',
    'tidy_json_schema',
]