import re
import inspect
import logging
import numpy as np

from annotated_types import MaxLen
from pydantic.fields import FieldInfo
from pydantic import Field, AliasChoices
from types import NoneType
from typing_extensions import TypeAliasType, ParamSpec, TypeVar
from typing import (no_type_check, TYPE_CHECKING, Type, Callable, Sequence, Union, Concatenate, TypeAlias, 
                    Generic)

from ...common_utils.type_utils import check_type_is
from ...common_utils.concurrent_utils import SyncOrAsyncFunc

from .index import (MilvusIndex, StringIndex, FloatVectorIndex, BinaryVectorIndex, ScalarIndex, VectorIndex)
from .query import QueryExpression, ExpressionT

if TYPE_CHECKING:
    from .base_model import MilvusModel
    from ...ai.chat import Prompt
    
_M = TypeVar('_M', bound='MilvusModel')
_T = TypeVar('_T', bound=Union[MilvusIndex, NoneType])

_VecFloatEmbeddingType: TypeAlias = Sequence[float] | np.ndarray
_MilvusEmbedderType_T1: TypeAlias = SyncOrAsyncFunc[["str|Prompt"], _VecFloatEmbeddingType]
_MilvusEmbedderType_T2 = TypeAliasType("_MilvusEmbedderType_T2", SyncOrAsyncFunc[[Type[_M], "str|Prompt"], _VecFloatEmbeddingType], type_params=(_M,))
MilvusEmbedderType = TypeAliasType("MilvusEmbedderType", Union[_MilvusEmbedderType_T1, _MilvusEmbedderType_T2], type_params=(_M,))
'''
Type hint for encoder function.
If vector index field is filled with encoder, you can pass text to `cls.Search` directly without
manually embed it.
'''

_logger = logging.getLogger(__name__)

class MilvusField(FieldInfo, Generic[_T, _M]):
    '''Milvus field'''
    
    name: str
    '''Field name. If not specified, the attribute name will be used'''
    is_pk: bool = False
    '''if this field is primary key'''
    auto_id: bool = False
    '''
    If this field is auto id. This only works on primary key field.
    Note: When key type=`int` and `auto_id` is True, the default value will be -1.
         Real ID will only be assigned after it is saved to DB.
    '''
    is_vector: bool = False
    '''Identify this field as vector field. If not, list/tuple/... will be see as array type'''
    is_partition_key: bool = False
    '''if this field is partition key'''
    dimension: int|None = None
    '''
    Specific dimension or size of the vector field, only works on vector.
    If `max_capacity` is given on a vector field, this field will be set to the value of `max_capacity`.
    '''
    max_capacity: int|None = None
    '''
    Specific dimension or size of the array field, only works on array type.
    If `dimension` is given on an array field, this field will be set to the value of `dimension`.
    '''
    index: _T|None = None
    '''Index of this field. If None, no index will be created on this field'''
    embedder: MilvusEmbedderType[_M]|None = None
    '''
    Embedding method to encode data to vector for searching.
    If this field is filled, you can pass text to `cls.Search` directly without 
    manually embed it.
    
    Note: This field is only available for vector fields.
    '''
    model_cls: 'Type[_M]|None' = None
    '''
    Model class binding to this field.
    This field will be updated when during model's initializing.
    '''
    non_milvus_field: bool = False
    '''
    If True, this field will not be treated as a milvus field, 
    and will not be created in milvus collection. This will be useful when you want to
    define some extra field which doesn't exist in milvus collection.
    '''
    
    @property
    def max_length(field: FieldInfo) -> int | None:
        '''
        Max length of the field. This only works for VARCHAR(string) field.
        For string fields, if max_length is not set, default len of 65535 will be used.
        '''
        for meta in field.metadata:
            if isinstance(meta, MaxLen):
                return meta.max_length
        return None
        
    def __class_getitem__(cls, item):
        return cls  # to prevent generic problem
    
    def __init__(
        self, 
        name: str="",
        is_pk:bool=False, 
        auto_id:bool=False, 
        is_vector:bool=False, 
        max_length:int|None=None,
        is_partition_key:bool=False, 
        dimension:int|None=None,
        max_capacity:int|None=None,
        index:_T|type[_T]|None=None,
        embedder: MilvusEmbedderType[_M]|None = None,
        model_cls: type[_M]|None = None, # just for filtering `model_cls` from `**kwargs`
        non_milvus_field: bool=False,
        **kwargs,
    ):
        if max_length is not None:
            kwargs['max_length'] = max_length
        if is_pk and auto_id:
            kwargs['default'] = None
        
        alias = kwargs.pop('alias', None)
        if alias and not name:
            name = alias
        
        validation_alias = kwargs.pop('validation_alias', None)
        serialization_alias = kwargs.pop('serialization_alias', None)
        if name:
            if not serialization_alias and not alias:
                serialization_alias = name
            
            if validation_alias:
                if isinstance(validation_alias, AliasChoices):
                    if name not in validation_alias.choices:
                        validation_alias.choices.append(name)
                else:
                    validation_alias = AliasChoices(name, validation_alias)
            else:
                validation_alias = AliasChoices(name)
            if alias and alias not in validation_alias.choices:
                validation_alias.choices.append(alias)
                alias = None
                
        super().__init__(
            alias=alias,
            serialization_alias=serialization_alias, 
            validation_alias=validation_alias,
            **kwargs
        )
        self.name = name    # will be updated later if `name` is an empty string
        self.is_pk = is_pk
        self.auto_id = auto_id
        self.is_vector = is_vector
        self.is_partition_key = is_partition_key
        self.max_capacity = max_capacity
        self.dimension = dimension
        self.non_milvus_field = non_milvus_field
        if isinstance(index, type):
            if issubclass(index, MilvusIndex):
                index = index()
            else:
                raise TypeError(f'Index type must be subclass of MilvusIndex, not {index}')
        self.index = index
        self.embedder = embedder
        
        if self.index:
            if isinstance(self.index, ScalarIndex) and self.is_vector:
                self.is_vector = False
                _logger.debug(f'Index is given in field `{name}` with is_vector=True. `is_vector` will be set to False. If this is not what you want, please set `is_vector` to False.')
            elif isinstance(self.index, VectorIndex) and not self.is_vector:
                self.is_vector = True
                _logger.debug(f'Index is given in field `{name}` with is_vector=False. `is_vector` will be set to True. If this is not what you want, please set `is_vector` to True.')
        
        if self.is_vector and not self.dimension and self.max_capacity:
            self.dimension = self.max_capacity
            _logger.debug(f'`max_capacity` is given in field `{name}` with is_vector=True. `max_capacity` will be set to `dimension`. If this is not what you want, please set `dimension` instead.')
        elif not self.is_vector and self.dimension and not self.max_capacity:
            self.max_capacity = self.dimension
            _logger.debug(f'`dimension` is given in field `{name}` with is_vector=False. `dimension` will be set to `max_capacity`. If this is not what you want, please set `max_capacity` instead.')
            
        if not is_vector and embedder:
            _logger.warning(f'Embedder is given in field `{name}` with is_vector=False. Embedder is only for vector field.')
    
    @no_type_check
    @classmethod
    def CastField(
        cls, 
        field: FieldInfo, 
        field_name: str="",
        model_cls: type[M]|None=None
    )->'MilvusField':
        '''When origin fieldinfo of pydantic is given, a new MilvusField will be created 
        by copying all attributes from the origin field'''
        if isinstance(field, MilvusField):
            field.model_cls = model_cls
            return field
        elif isinstance(field, FieldInfo):
            new_field = MilvusField()
            
            for attr in FieldInfo.__slots__:
                if attr.startswith('__') and attr.endswith('__'):
                    continue
                if isinstance(getattr(FieldInfo, attr), property):
                    continue
                setattr(new_field, attr, getattr(field, attr))
            
            new_field.name = field_name
            new_field.is_pk = False
            new_field.auto_id = False
            new_field.is_vector = False
            new_field.is_partition_key = False
            new_field.dimension = 0
            new_field.index = None
            new_field.embedder = None
            new_field.model_cls = model_cls
            new_field.default = field.default
            return new_field
        else:
            raise TypeError(f'Cannot cast {field} to MilvusField')

    def __eq__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, '==', other)
    
    def __ne__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, '!=', other)
    
    def __lt__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, '<', other)
    
    def __le__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, '<=', other)
    
    def __gt__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, '>', other)
    
    def __ge__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, '>=', other)
    
    def __and__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, 'and', other)
    
    def __or__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, 'or', other)
    
    def __xor__(self, other: ExpressionT):
        return QueryExpression[self.model_cls].Build(self, '^', other)
    
    def __invert__(self):
        return QueryExpression(f'not ({self})')

    def in_array(self, array: Sequence):
        '''
        Check if this field's value is in the given array
        e.g. `Model.x.in_array([1, 2, 3])`
        '''
        if isinstance(array, str):
            raise TypeError(f'Invalid array type, got `str`. It must be a Sequence(e.g. list).')
        array = list(array)
        if self.annotation:
            t = None
            if check_type_is(self.annotation, str):
                t = str
            elif check_type_is(self.annotation, int):
                t = int
            elif check_type_is(self.annotation, float):
                t = float
            if t:
                for i in range(len(array)):
                    array[i] = t(array[i])
        return QueryExpression[self.model_cls](f'{self.name} in {array}')

class _VectorField(MilvusField):
    def __init__(
        self, 
        dimension:int,
        index:FloatVectorIndex|BinaryVectorIndex|None = None,
        embedder: MilvusEmbedderType|None = None,
        **kwargs
    ):
        super().__init__(
            is_pk=False,
            auto_id=False, 
            is_vector=True, 
            is_partition_key=False, 
            dimension=dimension,
            index=index,
            embedder=embedder,
            **kwargs
        )

if TYPE_CHECKING:
    VectorField = TypeAliasType("VectorField", Union[MilvusField[FloatVectorIndex, _M], MilvusField[BinaryVectorIndex, _M]], type_params=(_M,))
else:
    VectorField = _VectorField
    
class ArrayField(MilvusField[NoneType, _M], Generic[_M]):
    def __init__(self, dimension:int, max_length:int=0, **kwargs):
        '''
        `max_length` is for defining the max length of VARCHAR(in case the sub type is str).
        For defining array's length, use `dimension` instead
        '''
        super().__init__(is_pk=False,
                         auto_id=False, 
                         is_vector=False, 
                         is_partition_key=False, 
                         dimension=dimension,
                         max_length=max_length,
                         **kwargs)
        
class StringField(MilvusField[StringIndex, _M], Generic[_M]):
    def __init__(self, max_length:int, index:StringIndex|None=None, **kwargs):
        '''String field must define `max_length`.'''
        super().__init__(is_pk=False,
                         auto_id=False, 
                         is_vector=False, 
                         is_partition_key=False, 
                         dimension=0,
                         max_length=max_length,
                         index=index,
                         **kwargs)
        
_source_files:dict[str, tuple[str, ...]] = {}   # type: ignore    

def _get_source_line(file_path:str, line: int)->str|None:
    if file_path not in _source_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = _source_files[file_path] = tuple(f.readlines())
    else:
        lines = _source_files[file_path]
    if line < 1 or line > len(lines):
        return None
    return lines[line-1].strip()

if TYPE_CHECKING:
    _R = TypeVar('_R')
    _P = ParamSpec('_P')
    _P2 = ParamSpec('_P2')
    
    def _func_param_extend(
        func: Callable[_P, _R], 
        name: str="",
        is_pk:bool=False, 
        auto_id:bool=False, 
        is_vector:bool=False, 
        max_length:int|None=None,
        is_partition_key:bool=False, 
        dimension:int|None=None,
        max_capacity:int|None=None,
        index:MilvusIndex|None=None,
        embedder: MilvusEmbedderType|None = None,
        non_milvus_field: bool=False,
        *args: _P.args, 
        **kwargs: _P.kwargs
    ) -> _R:
        '''
        All-in-one function for creating MilvusField.

        Args:
            name: name of this field in milvus. If your model field name is the same 
                    as the field name in milvus, you can leave it empty. 
                    This will also becomes an validation alias.
            is_pk: Whether the field is a primary key.
            auto_id: Whether the field is an auto id. This only works on primary key field.
            is_vector: Whether the field is a vector field.
            max_length: The maximum length of the field. 
            is_partition_key: Whether the field is a partition key.
            dimension: Specific dimension or size of the vector field, only works on vector.
                        If `max_capacity` is given on a vector field, this field will be set to the value of `max_capacity`.
            max_capacity: Specific dimension or size of the array field, only works on array type.
                            If `dimension` is given on an array field, this field will be set to the value of `dimension`.
            index: The index of the field. For string/vector field only. If index is scaler type but `is_vector` is True,
                    `is_vector` will be set to False. Same, If index is vector type but `is_vector` is False, `is_vector` will be set to True.
            embedder: Embedding method to encode data to vector for searching.
                        If this field is filled, you can pass text to `cls.Search` directly without
                        manually embed it. 
                        Note: This field is only available for vector fields.
            non_milvus_field: If True, this field will not be treated as a milvus field, 
                            and will not be created in milvus collection. This will be useful when you want to
                            define some extra field which doesn't exist in milvus collection.
            
        Kwargs(origin args for pydantic)
            default: The default value of the field.
            default_factory: The factory function used to construct the default for the field.
            alias: The alias name of the field.
            alias_priority: The priority of the field's alias.
            validation_alias: The validation alias of the field.
            serialization_alias: The serialization alias of the field.
            title: The title of the field.
            description: The description of the field.
            examples: List of examples of the field.
            exclude: Whether to exclude the field from the model serialization.
            discriminator: Field name or Discriminator for discriminating the type in a tagged union.
            deprecated: A deprecation message, an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport,
                or a boolean. If `True`, a default deprecation message will be emitted when accessing the field.
            json_schema_extra: A dict or callable to provide extra JSON schema properties.
            frozen: Whether the field is frozen.
            validate_default: Whether to validate the default value of the field.
            repr: Whether to include the field in representation of the model.
            init: Whether the field should be included in the constructor of the dataclass.
            init_var: Whether the field should _only_ be included in the constructor of the dataclass, and not stored.
            kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
            metadata: List of metadata constraints.
        '''
        ...
    
    def _wrap(f1: Callable[_P, _R], f2: Callable[Concatenate[Callable[_P, _R], _P2], _R])->Callable[_P2, _R]: ...

    field = _wrap(Field, _func_param_extend)    # type: ignore
else:
    def field(*,
            name: str="",
            is_pk:bool=False, 
            auto_id:bool=False, 
            is_vector:bool=False, 
            max_length:int|None=None,
            is_partition_key:bool=False, 
            dimension:int|None=None,
            max_capacity:int|None=None,
            index:MilvusIndex|None=None,
            embedder: MilvusEmbedderType|None = None,
            non_milvus_field: bool=False,
            **kwargs):
        
        alias = kwargs.get('alias', None)
        if alias and not name:
            name = alias
        
        # try to find the origin field name, if yes, add to validation alias
        frame = inspect.currentframe().f_back
        field_define_pattern = r'^[a-z_][a-z\d_]*\s*\:\s*[a-z_][\d\[\]a-z_]*\s*=\s*(?:[a-z][\da-z_]*\(.*\)?|\\)'
        model_field_name = None # the origin model field name, e.g. x: int = 1, x is the model field name
        while frame:
            file_path = frame.f_code.co_filename
            if not (file_path.startswith('<') and file_path.endswith('>')):
                source_code = _get_source_line(file_path, frame.f_lineno)
                if not source_code:
                    break
                
                if re.match(field_define_pattern, source_code, re.DOTALL|re.IGNORECASE):
                    model_field_name = source_code.split(':')[0].strip()
                    break
            frame = frame.f_back
            
        if model_field_name:
            if not name:
                name = model_field_name
            validation_alias = kwargs.pop('validation_alias', None)
            if validation_alias:
                if isinstance(validation_alias, AliasChoices):
                    if model_field_name not in validation_alias.choices:
                        validation_alias.choices.append(model_field_name)
                else:
                    validation_alias = AliasChoices(model_field_name, validation_alias)
            elif name != model_field_name:
                validation_alias = AliasChoices(model_field_name)
            kwargs['validation_alias'] = validation_alias
        
        # will keep checking if `name` in validation alias/serialization alias during __init__ of `MilvusField`
        return MilvusField(
            name=name, 
            is_pk=is_pk, 
            auto_id=auto_id, 
            is_vector=is_vector, 
            max_length=max_length,
            is_partition_key=is_partition_key, 
            dimension=dimension,
            index=index,
            max_capacity=max_capacity,
            embedder=embedder,
            non_milvus_field=non_milvus_field,
            **kwargs
        )


__all__ = ['MilvusEmbedderType', 'MilvusField', 'VectorField', 'ArrayField', 'StringField', 'field']