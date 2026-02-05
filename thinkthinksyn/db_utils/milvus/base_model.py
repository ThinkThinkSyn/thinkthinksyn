# -*- coding: utf-8 -*-
'''ORM base class of milvus'''

import os

if __name__ == "__main__": # for debugging
    import sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'thinkthinksyn.db_utils.milvus'

import re
import uuid
import asyncio
import logging
import hashlib
import pymilvus
import numpy as np

from enum import Enum
from types import NoneType
from abc import ABC, abstractmethod
from annotated_types import MaxLen
from typing import (Self, Optional, Sequence, overload, Any, TYPE_CHECKING, ClassVar, 
                    Coroutine, Literal, no_type_check, TypeAlias, TypeVar, ParamSpec)
from typing_extensions import override

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from pydantic._internal._model_construction import ModelMetaclass   
from pydantic import BaseModel, model_validator, Field as PDField, model_serializer, TypeAdapter

from pymilvus.grpc_gen.milvus_pb2 import MutationResult as GRPCMutationResult
from pymilvus import (connections, FieldSchema, CollectionSchema, DataType as Milvus_DataType, 
                      utility as milvus_utility, Index, Partition)
from pymilvus.client.types import LoadState
from pymilvus.client.asynch import MutationResult, MutationFuture
from pymilvus.client.abstract import Hit
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.grpc_handler import GrpcHandler

from ...common_utils.decorators import class_property
from ...common_utils.concurrent_utils import wait_coroutine
from ...common_utils.type_utils import (check_value_is, check_type_is, func_arg_count, get_pydantic_model_field_aliases, serialize)
from ...common_utils.network_utils import get_local_ip

from ._collection import Collection
from .index import _parse_anno    # must be imported explicitly
from .index import *
from .fields import *
from .query import QueryExpression

_logger = logging.getLogger(__name__)
_md5_hash = hashlib.md5()

def _hash_md5(s: str) -> str:
    _md5_hash.update(s.encode('utf-8'))
    return _md5_hash.hexdigest()

def _get_field_max_len(field: FieldInfo) -> int:
    '''get the max length of a field. If not found,
    default to be `65535` for varchar field'''
    for meta in field.metadata:
        if isinstance(meta, MaxLen):
            return meta.max_length
    return 65535

def _gen_uuid4():
    return str(uuid.uuid4()).replace('-', '')

def _cosine_similarity(v1:Sequence[float|int]|np.ndarray, 
                      v2:Sequence[float|int]|np.ndarray)->float:
    assert len(v1) == len(v2), 'vectors must have the same length'
    if isinstance(v1, Sequence):
        v1 = np.array(v1)
    if isinstance(v2, Sequence):
        v2 = np.array(v2)
    v1.reshape(1, -1)
    v2.reshape(-1, 1)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def _get_env(key, default=None)->str|None:
    return os.environ.get(key, default)

_default_milvus_host = _get_env('MILVUS_HOST', default='localhost')
_default_milvus_port = _get_env('MILVUS_PORT', default=19530)
_default_milvus_user = _get_env('MILVUS_USER', default=None)
_default_milvus_pw = _get_env('MILVUS_PASSWORD', default=None)
_default_milvus_token = _get_env('MILVUS_TOKEN', default=None)

class MilvusConnectionConfig(BaseModel):
    host: str = 'localhost'
    port: int = 19530
    user: str|None = None
    password: str|None = None
    token: str|None = None
    connection_alias: str|None = None
    
    def model_post_init(self, _):
        if self.host != 'localhost' and self.host != '127.0.0.1':
            if self.host == get_local_ip():
                self.host = 'localhost'
    
    def get_connection_alias(self, db: str) -> str:
        if self.connection_alias:
            return f'{self.connection_alias}:{db}'
        else:
            key = f'{self.host}:{self.port}:{self.user}:{self.password}:{self.token}'
            return f'{_hash_md5(key)}:{db}'
    
    def get_connection(self, db: str|None=None):
        db = db or 'default'
        if (conn:=getattr(self, f'_connection_{db}', None)) is None:
            alias = self.get_connection_alias(db)
            connections.connect(
                host=self.host,
                port=self.port,
                user=self.user or '',
                password=self.password or '',
                token=self.token or '',
                alias=alias,
                db_name=db
            )
            conn = connections._fetch_handler(alias)
            setattr(self, f'_connection_{db}', conn)
        return conn

    def copy(self, **update_fields):
        data = self.model_copy(update=update_fields)
        if update_fields:
            for attr in tuple(self.__dict__.keys()):
                if attr.startswith('_connection'):
                    delattr(data, attr)
        return data

_default_config = MilvusConnectionConfig(
    host=_default_milvus_host,  # type: ignore
    port=int(_default_milvus_port),  # type: ignore
    user=_default_milvus_user,
    password=_default_milvus_pw,
    token=_default_milvus_token,
    connection_alias='default'
)

_default_conn = None

_OtherSupportedModelTypes: TypeAlias = type['MilvusModel']

def _check_or_create_db(db: str, conn=None):
    global _default_conn
    if not conn:
        conn = _default_conn = _default_config.get_connection()
    if not db in conn.list_database():  # type: ignore
        conn.create_database(db)    # type: ignore

def _get_model_enum_fields(model: type[BaseModel])->dict[str, MilvusField]:
    if not '__model_enum_fields__' in model.__dict__:
        fields = dict()
        for k, v in model.model_fields.items():
            if v.annotation and isinstance(v.annotation, type) \
                and check_type_is(v.annotation, Enum):  # annotation should be a class 
                fields[k] = v
        model.__model_enum_fields__ = fields        # type: ignore
    return model.__model_enum_fields__      # type: ignore

def _is_vector_type(t, t_args)->bool:
    if check_type_is(t, (list, tuple, np.ndarray)):
        if check_type_is(t, list):
            if t_args and len(t_args)==1:
                if check_type_is(t_args[0], (float, int)):
                    return True
        elif check_type_is(t, tuple):
            if len(t_args) == 2:
                if t_args[1] is Ellipsis and check_type_is(t_args[0], (float, int)):
                    return True
        else:   # ndarray
            return True
    return False

def _get_enum_by_value(enum_cls: type[Enum], value: str)->Enum:
    for e in enum_cls:
        if e.value == value:
            return e
    raise Exception(f'Enum value not found by value: {value}')

def _get_orm_model_pk_type(model: _OtherSupportedModelTypes)->type:    
    if check_type_is(model, MilvusModel):
        pk_field = model.model_fields[model.__pk_field__]
        return pk_field.annotation  # type: ignore  
    else:
        raise Exception(f'Unknown db model type. Got: `{model}`. Fail to find the primary key type.')    
    
class MilvusModelMetaclass(ModelMetaclass):
    def __getattr__(self, name): 
        if not name.startswith('_') and name != 'model_fields':
            model_fields = getattr(self, 'model_fields', {})
            field_name_map = getattr(self, '__field_name_map__', {})
            if name in model_fields:
                return model_fields[name]
            elif name in field_name_map:
                return model_fields[field_name_map[name]]
        return super().__getattr__(name)    # type: ignore

class _EmptyMutationResult(MutationResult):
    '''special mutation result for no mutation case'''
    def __init__(self, raw: GRPCMutationResult|None = None):
        super().__init__(raw)   
    
    @override
    def _pack(self, raw):
        if raw is None:
            return
        super()._pack(raw)
        
class CombineMutationResult(_EmptyMutationResult):
    def __init__(self, *raw: GRPCMutationResult|None):
        super().__init__(None)
        self._raw = []
        for r in raw:
            self._pack(r._raw) if r else None       # type: ignore
            
    @override
    def _pack(self, raw):
        if raw is None:
            return
        which = raw.IDs.WhichOneof("id_field")
        if which == "int_id":
            self._primary_keys.extend(raw.IDs.int_id.data)
        elif which == "str_id":
            self._primary_keys.extend(raw.IDs.str_id.data)
        self._insert_cnt += raw.insert_cnt
        self._delete_cnt += raw.delete_cnt
        self._upsert_cnt += raw.upsert_cnt
        self._timestamp = max(self._timestamp, raw.timestamp)
        self._succ_index.extend(raw.succ_index)
        self._err_index.extend(raw.err_index)
        self._raw.append(raw)

class MilvusModel(BaseModel, ABC, metaclass=MilvusModelMetaclass):
    '''
    ORM of Milvus collection. Suitable for Milvus 2.3.5
    Model class args:
        db (str|Team): The db name of the model. It can be string or team(team name will be used in this case).
                       Default to be `default`, which is a common area for all teams. 
                       Note that if db==`global`, it will be directed to `default`.
        collection_name (str|None): the collection name of the model. If not specified, will use class name.
        alias:(str|None): The alias of the collection. If not specified, will use "{db}_{collection_name}.
        abstract(bool): In case you want to be an abstract base class to provide some common fields,
                        set this to True. Default to be False. Abstract model will not be created to Milvus.
        connection(MilvusConnectionConfig|AsyncGrpcHandler|GrpcHandler|None): The connection config of this model.
                          If not specified, will use default connection config.
                          
    Note: You could assign collection description by typing comment under model class
    
    Example 1:
    ```
    from utils.db.milvus import field, MilvusModel, FloatVectorIndex_IVF_FLAT
    
    class LawContent(MilvusModel, db='law_chapters', collection_name='law_overall_content_summarized_embedding'):
        id: int = field(is_pk=True, auto_id=True)
        chapID: str = field(max_length=6)
        scheduleID: str = field(max_length=5)
        partID:str = field(max_length=5)
        divisionID:str = field(max_length=5)
        subdivisionID:str = field(max_length=5)
        sectionID:str = field(max_length=10)
        subsectionID: str = field(max_length=50)
        paragraphID: str = field(max_length=50)
        subparagraphID: str = field(max_length=50)
        crossID: str = field(max_length=5)
        chunkID: str = field(max_length=5)
        original_content: str = field(max_length=10000)
        summarized_content: str = field(max_length=10000)
        summarized_content_embedding: list[float] = field(dimension=1024, is_vector=True, index=FloatVectorIndex_IVF_FLAT('summarized_content_embedding', 
                                                            nlist=1024, metric_type='COSINE'))
    ```
    
    Example 2:
    ```
    from utils.db.milvus import field, MilvusModel, FloatVectorIndex_IVF_FLAT
    
    class Test(MilvusModel, db='default', collection_name='test'):
        id: int = field(is_pk=True, auto_id=True,)
        a: int = field()
        b: str = field(max_length=10)
        c: list[float] = field(dimension=10, is_vector=True, index=FloatVectorIndex_IVF_FLAT())
    
    test = Test(a=1, b='2', c=[1 for i in range(10)])
    test.save()
    ```
    
    # TODO: embed another milvus model into a field
    '''
    
    # region internal fields
    model_config = {
        'arbitrary_types_allowed': True,    # to make BaseModel accept numpy types
        'use_attribute_docstrings': True    # this generate doc for each field by looking to the field's docstring
    }
    __collection__: ClassVar[Collection]
    '''The origin milvus collection client of this model'''
    __schema__: ClassVar[dict[str, FieldSchema]]
    '''
    All fields schema of this collection. 
    Note: the key is not exactly the field name if you has manually set in field info.
    '''
    __indices__: ClassVar[dict[str, Index]]
    '''The pymilvus indices of this collection'''
    __pk_field__: ClassVar[str]
    '''
    The primary key field name of this collection.
    Note: This is not real field name of the pk field. For real field name,
          please get by cls.model_fields[cls.__pk_field__].name  
    '''
    __db_name__: ClassVar[str]
    '''The db name of this collection'''
    __alias__: ClassVar[str]
    '''The alias of this collection. For connection'''
    __is_abstract__: ClassVar[bool]
    '''Whether this model is abstract'''
    __field_name_map__: ClassVar[dict[str, str]]
    '''
    Mapping the model field name to real field name in Milvus(in case you set it manually in field info).
    {real field name: field name}
    '''
    __field_alias_dict__: ClassVar[dict[str, set[str]]]
    '''{field name: field alias}'''
    __optional_str_fields__: ClassVar[set[str]]
    '''The optional string fields of this model'''
    __non_standard_type_adapters__: ClassVar[dict[str, TypeAdapter]]
    '''
    {field_name: pydantic TypeAdapter}.
    These adapters are for serializing non-standard field types.
    '''
    __other_db_fields__: ClassVar[dict[str, tuple[_OtherSupportedModelTypes, bool]]]
    '''
    Fields which has the type from another database ORM model.
    {field_name: (model_cls, is_optional)}
    '''
    
    if TYPE_CHECKING:
        model_fields: dict[str, MilvusField[MilvusIndex, Self]|MilvusField[NoneType, Self]]     
        '''
        The pydantic field infos of this model. Note that all fields are milvus fields, which is a subclass of pydantic FieldInfo.
        Note: the key is not exactly the real field name in milvus(if you has manually set in field info).
        '''
    # endregion
    
    # region extra model configs
    IsDynamic: ClassVar[bool] = False
    '''
    Whether to enable `enable_dynamic_field` to be True in milvus. 
    In that case, fields which is not pre-defined in schema will also be saved during insert. 
    '''
    # endregion
    
    extra_fields: dict[str, Any]|None = PDField(default=None, exclude=True)
    '''
    Extra fields of this record. This is only available when `IsDynamic` is True, 
    i.e. when cls.IsDynamic is False, this field will always be None.
    
    Note: This field will not be dump during serialization. In case cls.IsDynamic is True and
          this field contains values, key-val pairs will be dumped to the dict directly.
    '''
    
    # region magic methods
    def __init_subclass__(
        cls, 
        db: str|None = None,
        collection_name: str|None = None,
        alias: str|None = None,
        abstract: bool = False,
        connection: MilvusConnectionConfig|AsyncGrpcHandler|GrpcHandler|None = None,
        **kwargs
    ):
        '''For enable __pydantic_init_subclass__ & type hinting'''
        super().__init_subclass__(**kwargs)
    
    def __init__(self, **kwargs):
        no_pk = (self.RealPrimaryFieldName not in kwargs) and (self.__pk_field__ not in kwargs)
        if no_pk and self.IsAutoPrimaryKey and check_type_is(self.PrimaryKeyType, str):
            kwargs[self.__pk_field__] = _gen_uuid4()
        super().__init__(**kwargs)
        
    def __eq__(self, other):
        if super().__eq__(other):
            return True
        if isinstance(other, MilvusModel):
            if not self.IsAbstract and not other.IsAbstract:
                other_alias = getattr(other, '__alias__', None) 
                self_alias = getattr(self, '__alias__', None)
                if other_alias and self_alias and other_alias == self_alias:    
                    # pk checking can only be done when 2 connection alias is equivalent 
                    self_pk = self.pk
                    other_pk = other.pk
                    if self_pk is None or other_pk is None:
                        return False
                    if self.IsAutoPrimaryKey and check_type_is(self.PrimaryKeyType, int):
                        if self_pk == -1 or other_pk == -1:
                            return False
                    return self_pk == other_pk
        return False
    # endregion
    
    # region internal methods
    @model_validator(mode='before')
    @classmethod
    def _BaseMongoModelPreValidator(cls, data):
        if isinstance(data, dict):
            tidied_data = {}
            model_enum_fields = _get_model_enum_fields(cls)
            other_db_model_fields = cls.__other_db_fields__
            special_adapters = cls.__non_standard_type_adapters__
            
            if cls.IsDynamic:
                if 'extra_fields' in data:
                    extra_fields = data.pop('extra_fields')
                    if not isinstance(extra_fields, dict):
                        raise Exception('Extra fields must be dict')
                else:
                    extra_fields = {}
                tidied_data['extra_fields'] = extra_fields
            
            for key, val in data.items():
                if (real_key := cls.ContainsField(key)):
                    field_info = cls.model_fields[real_key]
                    if field_info.exclude:
                        # exclude fields, no need to check
                        tidied_data[real_key] = val
                    else:
                        # optional string
                        if real_key in cls.__optional_str_fields__ and val == '':
                            tidied_data[real_key] = None  # change empty string to `None` for optional string fields
                        
                        # special type adapters
                        elif real_key in special_adapters:
                            adapter = special_adapters[real_key]
                            if not check_value_is(val, field_info.annotation):
                                if isinstance(val, str):
                                    val = adapter.validate_json(val)
                            tidied_data[real_key] = val
                        
                        # enum
                        elif (
                            bool(model_enum_fields) and 
                            (real_key in model_enum_fields) and 
                            isinstance(val, str)
                        ):
                            tidied_data[real_key] = _get_enum_by_value(model_enum_fields[real_key].annotation, val)     # type: ignore         
                        
                        # db ORM fields
                        elif (
                            bool(other_db_model_fields) and
                            (real_key in other_db_model_fields) and
                            not check_value_is(val, other_db_model_fields[real_key][0])
                        ):
                            model_type, is_optional = other_db_model_fields[real_key]
                            if is_optional and val in ("", None, -1):
                                tidied_data[real_key] = None
                            # milvus
                            elif check_type_is(model_type, MilvusModel):
                                expr = getattr(model_type, model_type.__pk_field__).__eq__(val)
                                ins = model_type.FindOne(expr=expr)
                                if not ins:
                                    raise Exception(f'Fail to find record by primary key: {val} in Milvus Model {model_type}')
                                tidied_data[real_key] = ins
                            else:
                                raise Exception(f'Unknown db model type. Got: `{model_type}`')
                        else:
                            # normal fields
                            tidied_data[real_key] = val
                elif cls.IsDynamic:
                    # only dynamic model can have extra fields
                    extra_fields[key] = val
            
            no_pk = (cls.RealPrimaryFieldName not in tidied_data) and (cls.__pk_field__ not in tidied_data)
            if no_pk and cls.IsAutoPrimaryKey and check_type_is(cls.PrimaryKeyType, str):
                tidied_data[cls.__pk_field__] = _gen_uuid4()
                
            if no_pk and cls.IsAutoPrimaryKey:
                if check_type_is(cls.PrimaryKeyType, str):
                    tidied_data[cls.__pk_field__] = _gen_uuid4()
                elif check_type_is(cls.PrimaryKeyType, int):
                    tidied_data[cls.__pk_field__] = -1
            
            return tidied_data
        return data
    
    @model_serializer(mode='wrap')
    def _serializer(self, origin_serializer):
        data = origin_serializer(self)
        if isinstance(data, dict):
            if self.IsDynamic:
                data.pop('extra_fields', None)  # actually not needed
                if self.extra_fields:
                    data.update(self.extra_fields)  # add extra fields to the dict directly
            
            for field_name, field_info in self.__class__.model_fields.items():
                if field_name in data:
                    if field_name in self.__optional_str_fields__:
                        val = data.pop(field_name)
                        if val is None:
                            val = ''    # turn None to empty string
                        data[field_info.name] = val
                    elif field_name != field_info.name:
                        # change field name to real field name
                        data[field_info.name] = data.pop(field_name)    
            
            if self.IsAutoPrimaryKey and check_type_is(self.PrimaryKeyType, int):
               data.pop(self.__pk_field__, None)    # remove id field if it is auto int pk
        return data
    
    @classmethod
    @no_type_check
    def BuildMilvusFieldSchema(cls, real_field_name:str, field:MilvusField)->FieldSchema:
        '''
        Return params for pymilvus's FieldSchema constructor.
        Note: for any other unsupported field types, it will be converted into string.
        '''
        origin, args, is_optional = _parse_anno(field.annotation)
        
        # checking
        if field.index is not None:
            if isinstance(field.index, VectorIndex) and not field.is_vector:
                field.is_vector = True
                _logger.debug('You are assigning a vector index to non-vector field. Field is converted to vector field. If this is not what you want, please remove the index.')
            elif isinstance(field.index, ScalarIndex) and field.is_vector:
                field.is_vector = False
                _logger.debug('You are assigning a scalar index to vector field. Field is converted to scalar field. If this is not what you want, please remove the index.')
            field.index.validate_field(field)    
        
        if not field.is_pk and field.auto_id:
            _logger.warning('auto_id is ignored because this field is not primary key')
        if field.is_pk and not ((check_type_is(origin, int) or origin == np.int64) or check_type_is(origin, str)):
            raise Exception('Milvus\'s primary key must be integer or string')
        if field.is_pk and field.auto_id:
            if check_type_is(origin, str):
                _logger.info('String auto key is enabled. Please note that this function is done by this module, but not a origin functionality of Milvus. For each new instance, a random UUID will be generated as primary key.')
            elif not (check_type_is(origin, int) or origin == np.int64):
                raise Exception('auto_id is only available on integer primary key')
        if field.is_vector and not check_type_is(origin, (list, tuple, np.ndarray)):
            raise Exception('Vector field must be list/tuple/ndarray/...')
        if field.is_partition_key and not (check_type_is(origin, (str, int)) or origin == np.int64):
            raise Exception('Partition key must be string or integer')
        
        params = {
            'description': field.description,
            'is_primary': field.is_pk,
            'auto_id': field.auto_id and field.is_pk,
            'is_partition_key': field.is_partition_key,
        }
        if not is_optional and check_type_is(origin, str) and field.is_pk and field.auto_id:
            params['auto_id'] = False    # Milvus has no auto string pk, this functionality is done by this module
        
        if field.default != PydanticUndefined and field.default is not None:
            if not check_value_is(field.default, field.annotation): 
                raise Exception(f'Default value type mismatch on field "{real_field_name}". Expecting {field.annotation}, but got {type(field.default)}')
            params['default_value'] = field.default
        
        origin_field_name = cls.ContainsField(real_field_name)
        if origin_field_name in cls.__other_db_fields__:
            pk_type = _get_orm_model_pk_type(cls.__other_db_fields__[origin_field_name][0])
            if isinstance(pk_type, int):
                params['dtype'] = Milvus_DataType.INT64
            else:
                params['dtype'] = Milvus_DataType.VARCHAR
                params['max_length'] = 65535
        elif check_type_is(origin, str):
            params['dtype'] = Milvus_DataType.VARCHAR
            max_len = field.max_length
            if max_len is None:
                max_len = 65535
            params['max_length'] = max_len
        elif not is_optional:
            if check_type_is(origin, (dict, BaseModel)):
                # TODO: check json available, if not, convert to string
                params['dtype'] = Milvus_DataType.JSON
            elif check_type_is(origin, int) or origin == np.int64:
                params['dtype'] = Milvus_DataType.INT64
            elif origin == np.int32:
                params['dtype'] = Milvus_DataType.INT32
            elif origin == np.int16:
                params['dtype'] = Milvus_DataType.INT16
            elif origin == np.int8:
                params['dtype'] = Milvus_DataType.INT8
            elif check_type_is(origin, float):
                params['dtype'] = Milvus_DataType.FLOAT
            elif origin == np.float64:
                params['dtype'] = Milvus_DataType.DOUBLE
            elif origin in (np.float32, np.float16):
                params['dtype'] = Milvus_DataType.FLOAT
            elif check_type_is(origin, bool):
                params['dtype'] = Milvus_DataType.BOOL
            elif check_type_is(origin, Enum):
                params['dtype'] = Milvus_DataType.VARCHAR
            elif _is_vector_type(origin, args):
                if len(args) != 1:
                    raise Exception('Vector field must have only one type argument, e.g. str/float/int')
                sub_type = args[0]
                if (field.dimension is None or field.dimension <=0) and field.is_vector:
                    if not check_type_is(sub_type, str) and field.max_length:
                        field.dimension = field.max_length
                        _logger.warning('You are not specifying dimension for vector field, but `max_length` is detected. Dimension is set to max_length, but it is recommended to specify dimension explicitly.')
                    else:
                        raise Exception(f'Incorrect dimension, {field} got:{field.dimension}. You must specify dimension for vector/array field. Use MilvusField(dimension=...)')
                
                if field.is_vector:
                    if check_type_is(sub_type, int) or sub_type in (np.int64, np.int32, np.int16, np.int8):
                        _logger.warning('You are assigning a int vector field which is not supported by milvus. Vector field is converted to float vector')
                    if not _logger.check_type_is(sub_type, float) and sub_type not in (np.float64, np.float32, np.float16):
                        raise Exception('Vector field must be float/int. Currently not supporting binary array.')
                    params['dtype'] = Milvus_DataType.FLOAT_VECTOR
                    params['dim'] = field.dimension
                else:
                    params['dtype'] = Milvus_DataType.ARRAY
                    params['max_capacity'] = field.max_capacity
                    if check_type_is(sub_type, int) or sub_type == np.int64:
                        params['element_type'] = Milvus_DataType.INT64
                    elif sub_type == np.int32:
                        params['element_type'] = Milvus_DataType.INT32
                    elif sub_type == np.int16:
                        params['element_type'] = Milvus_DataType.INT16
                    elif sub_type == np.int8:
                        params['element_type'] = Milvus_DataType.INT8
                    elif check_type_is(sub_type, float) or sub_type in (np.float64, np.float32, np.float16):
                        params['element_type'] = Milvus_DataType.FLOAT
                    elif check_type_is(sub_type, bool):
                        params['element_type'] = Milvus_DataType.BOOL
                    elif check_type_is(sub_type, str):
                        params['element_type'] = Milvus_DataType.VARCHAR
                        max_len = field.max_length
                        if max_len is None:
                            max_len = 65535
                        params['max_length'] = max_len
                    else:
                        raise Exception('Array sub-type must be int/float/bool/string')
            else:
                # default to be string
                params['dtype'] = Milvus_DataType.VARCHAR
                params['max_length'] = 65535
        else:
            # all other types are converted to string
            params['dtype'] = Milvus_DataType.VARCHAR   
            params['max_length'] = 65535
        
        _logger.debug(f'({cls.ClassName}.BuildMilvusFieldSchema) Built milvus schema for standard field `{field.name}`: {params}')
        return FieldSchema(name=real_field_name, **params)
    
    @classmethod
    def BuildMilvusCollectionSchema(cls) -> CollectionSchema:
        '''build milvus collection schema of this model class'''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        milvus_fields = list[FieldSchema]()
        for field_name, field in cls.model_fields.items():
            if not isinstance(field, MilvusField):
                field = MilvusField.CastField(field, field_name, model_cls=cls)
            else:
                if not field.name:
                    field.name = field_name
            if field.exclude or field.non_milvus_field:
                continue    # exclude field wil not be included in schema
                    
            real_field_name = field.name
            if field_name in cls.__non_standard_type_adapters__:
                # non-standard field type
                params = {
                    'description': field.description,
                    'is_primary': field.is_pk,
                    'auto_id': field.auto_id and field.is_pk,
                    'is_partition_key': field.is_partition_key,
                    'dtype': Milvus_DataType.VARCHAR,
                    'max_length': 65535,
                }
                if field.default != PydanticUndefined and field.default is not None:
                    if not isinstance(field.default, str):
                        milvus_default_val = cls.__non_standard_type_adapters__[field_name].dump_json(field.default).decode('utf-8')
                    else:
                        milvus_default_val = field.default
                    params['default_value'] = milvus_default_val
                _logger.debug(f'({cls.ClassName}.BuildMilvusCollectionSchema) Built milvus schema for non-standard field `{field_name}`: {params}')
                schema = FieldSchema(name=real_field_name, **params)    
            else:
                schema = cls.BuildMilvusFieldSchema(real_field_name, field)
            
            milvus_fields.append(schema)
        
        pk_valid = False
        vector_field_count = 0
        for field in milvus_fields:
            if field.is_primary:
                pk_valid = True # there is primary key
            elif field.dtype in (Milvus_DataType.FLOAT_VECTOR, Milvus_DataType.BINARY_VECTOR):
                vector_field_count += 1
        
        if not pk_valid:
            raise Exception('There is no primary key field. Currently Milvus not supporting collection without primary key.')
        if vector_field_count != 1:
            if vector_field_count == 0:
                _logger.warning(f'No vector field found in this model. Each Milvus collection should have one vector field.')
            elif vector_field_count !=1:
                if pymilvus.__version__ < '2.4':
                    raise Exception(f'Only milvus with version >2.4 support multiple vector fields. Current version = {pymilvus.__version__}, but got {vector_field_count} vector fields.')
        return CollectionSchema(fields=milvus_fields, description=cls.__doc__ or "", enable_dynamic_field=cls.IsDynamic)
    
    @classmethod
    @overload
    def CreateIndices(cls, collection: Collection|None=None, sync=True): ...
    @classmethod
    @overload
    async def CreateIndices(cls, collection: Collection|None=None, sync=True, _async=True): ...
    @classmethod
    def CreateIndices(cls, collection: Collection|None=None, sync=True, _async=False):
        '''create indices by looking to model's fields(MilvusField)'''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not collection:
            collection = cls.__collection__
        fields:dict[str, MilvusField] = cls.model_fields    
        if _async:
            async def wrapper(collection, sync):
                status = []
                for field_name, field_info in fields.items():
                    if field_info.index is not None:
                        index: MilvusIndex = field_info.index
                        create_params = index.build_create_index_params(field_info.name)
                        for k in ('name', 'field_name'):
                            if k in create_params:
                                del create_params[k]
                        index_name = create_params.pop('index_name', f'{field_name}_index')
                        real_field_name = field_info.name
                        s = await collection.create_index(real_field_name, create_params, index_name=index_name, sync=sync, _async=True)
                        status.append(s)
                return status
            return wrapper(collection, sync)
        else:
            status = []
            for field_name, field_info in fields.items():
                if field_info.index is not None:
                    index: MilvusIndex = field_info.index
                    create_params = index.build_create_index_params(field_info.name)
                    for k in ('name', 'field_name'):
                        if k in create_params:
                            del create_params[k]
                    index_name = create_params.pop('index_name', f'{field_name}_index')
                    s = collection.create_index(field_info.name, create_params, index_name=index_name, sync=sync)
                    status.append(s)
            return status
        
    @classmethod
    def CheckSchema(cls, field: MilvusField, milvus_schema: FieldSchema):
        '''
        Check whether pydantic schema is compatible with milvus schema(assume name is equal).
        For some attributes, if the given value is not equal to the value you have set, it will be forced 
        to align with the correct one in milvus. 
        '''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        field_name = milvus_schema.name
        py_type, py_type_args, is_optional = _parse_anno(field.annotation)
        milvus_schema_data_type = milvus_schema.dtype
        
        if field.is_pk != milvus_schema.is_primary:
            field.is_pk = milvus_schema.is_primary
            if field.is_pk:
                cls.__pk_field__ = field_name
                _logger.warning(f'Primary key field "{field_name}" is forced to be primary key')
            else:
                _logger.warning(f'Primary key field "{field_name}" is forced to be non-primary key')
                
        if field.auto_id != milvus_schema.auto_id:
            if not check_type_is(py_type, str):
                field.auto_id = milvus_schema.auto_id
                _logger.warning(f'Auto id mismatch on field "{field_name}", got "{field.auto_id}" and "{milvus_schema.auto_id}". It will be forced to be "{milvus_schema.auto_id}"')
            
        if field.is_partition_key != milvus_schema.is_partition_key:
            field.is_partition_key = milvus_schema.is_partition_key
            _logger.warning(f'Partition key mismatch on field "{field_name}", got "{field.is_partition_key}" and "{milvus_schema.is_partition_key}". It will be forced to be "{milvus_schema.is_partition_key}"')
        
        pydantic_default = field.default
        if pydantic_default is PydanticUndefined:
            pydantic_default = None
            
        milvus_default = getattr(milvus_schema, 'default_value', None)
        if pydantic_default is not None:
            pydantic_default = serialize(pydantic_default)
        
        if milvus_default:
            milvus_default = re.split("^.*data: ", str(milvus_default))[-1].strip() # e.g. "long_data: 1"
            
        if pydantic_default != milvus_default:
            _logger.warning(f'({cls.ClassName}.CheckSchema) Default value mismatch on field "{field_name}", got "{pydantic_default}", but milvus\'s default is "{milvus_default}".')
        
        if check_type_is(py_type, str):
            if milvus_schema_data_type != Milvus_DataType.VARCHAR:
                raise Exception(f'Data type mismatch on field "{field_name}"')
            if _get_field_max_len(field) != milvus_schema.max_length:
                raise Exception(f'Max length mismatch on field "{field_name}"')
        
        elif not is_optional:
            if check_type_is(py_type, int) or py_type == np.int64:
                if milvus_schema_data_type != Milvus_DataType.INT64:
                    raise Exception(f'Data type mismatch on field "{field_name}"')
            elif py_type == np.int32:
                if milvus_schema_data_type != Milvus_DataType.INT32:
                    raise Exception(f'Data type mismatch on field "{field_name}"')
            elif py_type == np.int16:
                if milvus_schema_data_type != Milvus_DataType.INT16:
                    raise Exception(f'Data type mismatch on field "{field_name}"')
            elif py_type == np.int8:
                if milvus_schema_data_type != Milvus_DataType.INT8:
                    raise Exception(f'Data type mismatch on field "{field_name}"')
            elif check_type_is(py_type, float) or py_type in (np.float64, np.float32, np.float16):
                if milvus_schema_data_type != Milvus_DataType.FLOAT:
                    raise Exception(f'Data type mismatch on field "{field_name}"')
            elif py_type == bool:
                if milvus_schema_data_type != Milvus_DataType.BOOL:
                    raise Exception(f'Data type mismatch on field "{field_name}"')
            elif _is_vector_type(py_type, py_type_args):
                if field.is_vector:
                    if milvus_schema_data_type != Milvus_DataType.FLOAT_VECTOR:
                        raise Exception(f'Data type mismatch on field "{field_name}"')
                    if field.dimension != milvus_schema.dim:
                        if not field.dimension:
                            field.dimension = milvus_schema.dim
                        else:
                            raise Exception(f'Dimension mismatch on field "{field_name}"')
                    # TODO: binary vector
                else:
                    if milvus_schema_data_type != Milvus_DataType.ARRAY:
                        raise Exception(f'Data type mismatch on field "{field_name}"')

                    milvus_schema_max_capacity = milvus_schema.max_capacity
                    if isinstance(milvus_schema_max_capacity, int):
                        pass
                    elif isinstance(milvus_schema_max_capacity, str):
                        milvus_schema_max_capacity = int(milvus_schema_max_capacity)
                    else:
                        raise Exception(f'Unknown milvus max capacity type: {type(milvus_schema_max_capacity)}')
                
                    if field.max_capacity != milvus_schema_max_capacity:
                        if field.max_capacity is None:
                            field.max_capacity = milvus_schema_max_capacity
                        else:
                            raise Exception(
                                f'Max capacity mismatch on field "{field_name}". Expecting {milvus_schema_max_capacity}, but got {field.max_capacity}'
                            )
                    sub_type = py_type_args[0] if py_type_args else None
                    if sub_type:
                        if sub_type == str:
                            if milvus_schema.element_type != Milvus_DataType.VARCHAR:
                                raise Exception(f'Data type mismatch on field "{field_name}"')
                            # ! these milvus schema fields are in string for some reason
                            milvus_schema_max_length = milvus_schema.max_length
                            if isinstance(milvus_schema_max_length, str):
                                milvus_schema_max_length = int(milvus_schema_max_length)
                            if _get_field_max_len(field) != milvus_schema_max_length:
                                raise Exception(f'Max length mismatch on field "{field_name}"')
                        elif check_type_is(sub_type, int) or sub_type == np.int64:
                            if milvus_schema.element_type != Milvus_DataType.INT64:
                                raise Exception(f'Data type mismatch on field "{field_name}"')
                        elif sub_type == np.int32:
                            if milvus_schema.element_type != Milvus_DataType.INT32:
                                raise Exception(f'Data type mismatch on field "{field_name}"')
                        elif sub_type == np.int16:
                            if milvus_schema.element_type != Milvus_DataType.INT16:
                                raise Exception(f'Data type mismatch on field "{field_name}"')
                        elif sub_type == np.int8:
                            if milvus_schema.element_type != Milvus_DataType.INT8:
                                raise Exception(f'Data type mismatch on field "{field_name}"')
                        elif check_type_is(sub_type, float) or sub_type in (np.float64, np.float32, np.float16):
                            if milvus_schema.element_type != Milvus_DataType.FLOAT:
                                raise Exception(f'Data type mismatch on field "{field_name}"')
                        elif check_type_is(sub_type, bool):
                            if milvus_schema.element_type != Milvus_DataType.BOOL:
                                raise Exception(f'Data type mismatch on field "{field_name}"')
            
    @classmethod
    def CheckIndex(cls, index: MilvusIndex, milvus_index: Index):
        '''Check if pydantic index is compatible with milvus index(assume name is equal)'''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        index.validate_index(milvus_index.params)
    
    @classmethod
    def _GetPkFieldName(cls, schema:CollectionSchema):
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        for field in schema.fields:
            if field.is_primary:
                return field.name
        return None
    
    @classmethod
    def ContainsField(cls, name_or_alias: str)->str|None:
        '''
        Check wether the given field name or field alias represents
        a field in this model.
        If yes, return the formal field name, otherwise return None.
        '''
        if name_or_alias in cls.model_fields:
            return name_or_alias
        if not hasattr(cls, '__field_alias_dict__'):
            cls.__field_alias_dict__ = {}
            for fn, f in cls.model_fields.items():      # type: ignore
                cls.__field_alias_dict__[fn] = set(get_pydantic_model_field_aliases(cls, fn))
                if f.name:
                    cls.__field_alias_dict__[fn].add(f.name)
                    
        for origin_name, field_alias in cls.__field_alias_dict__.items():
            if name_or_alias in field_alias:
                return origin_name
        return None
    
    @class_property
    def ClassName(cls):
        '''return the name of this model, e.g. `class A` -> 'A`'''
        return cls.__qualname__.split('.')[-1]
    
    @class_property
    def IsAbstract(cls):
        return cls.__is_abstract__
    
    @class_property
    def DBName(cls):
        return cls.__db_name__
    
    @class_property
    def ConnectionAlias(cls)->str:
        if cls.IsAbstract:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not hasattr(cls, '__alias__'):
            raise Exception(f'Alias of {cls.__name__} is not defined')
        return cls.__alias__
    
    @class_property
    def CollectionName(cls)->str:
        if cls.IsAbstract:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not hasattr(cls, '__collection__'):
            raise Exception(f'Collection of {cls.__name__} is not defined')
        return cls.__collection__.name
    
    @class_property
    def LoadState(cls) -> LoadState:
        '''Get the load state of this collection'''
        if cls.__is_abstract__:
            raise Exception(f'Load State of abstract class {cls.__name__} is undefined')
        connection = cls.__collection__._get_connection()
        return connection.get_load_state(cls.__collection__.name)
    
    @class_property
    def IsLoaded(cls) -> bool:
        '''Check whether this collection is loaded'''
        return cls.LoadState == LoadState.Loaded
    
    @classmethod
    def __pydantic_init_subclass__(
        cls, 
        db:str|None = None,    # type: ignore
        collection_name:str|None=None, 
        alias:str|None=None,
        abstract:bool=False,
        connection: MilvusConnectionConfig|AsyncGrpcHandler|GrpcHandler|None = None,
        **kwargs,
    ):
        '''An abstract class will be initialized when no collection name is passed'''
        if not collection_name and (c:=kwargs.pop('collection', None)) is not None:
            if isinstance(c, str):
                # `collection` is an alias for `collection_name`
                collection_name = c
        super().__pydantic_init_subclass__(**kwargs)    # type: ignore
        
        if not db:
            super_cls = cls.__bases__[0]        # type: ignore
            if super_cls != BaseModel and hasattr(super_cls, '__db_name__') \
                and super_cls.__db_name__ and isinstance(super_cls.__db_name__, str):
                db: str = super_cls.__db_name__     
            else:
                db = 'default'
        if db.lower() in ['default', 'global']:
            db = 'default'  # default & global will be directed to the default db
        
        if not collection_name:
            collection_name = cls.__qualname__.split('.')[-1]  # type: ignore     
        
        if connection:
            if isinstance(connection, MilvusConnectionConfig):
                if not alias:
                    alias = connection.get_connection_alias(db)
                connection = connection.get_connection(db)
            else:
                if not alias:
                    alias = f'{str(id(connection))}:{db}'
                connections._connected_alias[alias] = connection
                connections._alias[alias] = {}
        else:
            if not alias:
                alias = _default_config.get_connection_alias(db)
            connection = _default_config.get_connection(db)
            
        _check_or_create_db(db, connection) # if db not exists, create it
        
        cls.__alias__ = alias
        cls.__is_abstract__ = abstract
        cls.__db_name__ = db
        
        cls.__field_name_map__ = {}
        for field_name, field in tuple(cls.model_fields.items()):   # type: ignore
            if not isinstance(field, MilvusField):
                field = MilvusField.CastField(field, field_name, model_cls=cls)
                cls.model_fields[field_name] = field
            else:   # is `MilvusField`
                field.model_cls = cls   
                if not field.name:
                    field.name = field_name
                
            if not field.exclude and not getattr(field, 'non_milvus_field', False):
                cls.__field_name_map__[field.name] = field_name
        
        cls.__field_alias_dict__ = {}
        for field_name, field in cls.model_fields.items():      # type: ignore
            cls.__field_alias_dict__[field_name] = set(get_pydantic_model_field_aliases(cls, field_name))
            if field.name:
                cls.__field_alias_dict__[field_name].add(field.name)
        
        cls.__optional_str_fields__ = set()
        cls.__non_standard_type_adapters__ = {}
        cls.__other_db_fields__ = {}
    
        for field_name, field in cls.model_fields.items():      # type: ignore
            if field.exclude or getattr(field, 'non_milvus_field', False):   # e.g. `extra_fields`
                continue
            anno = field.annotation
            field_type, field_args, field_is_optional = _parse_anno(anno)
            if check_type_is(field_type, MilvusModel):
                cls.__other_db_fields__[field_name] = (field_type, field_is_optional)
            elif check_type_is(field_type, str):
                if field_is_optional:
                    cls.__optional_str_fields__.add(field_name)
            elif not field_is_optional:
                acceptable_types = (dict, BaseModel, int, float, bool, Enum, list, tuple,
                                    np.ndarray, np.number)
                acceptable_np_types = (np.int64, np.int32, np.int16, np.int8, np.float64, np.float32, np.float16)
                if not check_type_is(anno, acceptable_types) and (anno not in acceptable_np_types):   
                    # non-standard field type
                    adapter = TypeAdapter(anno)
                    cls.__non_standard_type_adapters__[field_name] = adapter
                elif check_type_is(anno, (tuple, list)) and \
                    not _is_vector_type(field_type, field_args):
                    # non-standard field type
                    adapter = TypeAdapter(anno)
                    cls.__non_standard_type_adapters__[field_name] = adapter
            else:   # non-standard field type
                adapter = TypeAdapter(anno)
                cls.__non_standard_type_adapters__[field_name] = adapter
        
        if not abstract:
            cls.__is_abstract__ = False
            
            if milvus_utility.has_collection(collection_name=collection_name, using=alias):     # type: ignore
                _logger.debug(f'({cls.ClassName}.pydantic_init_subclass) Collection "{collection_name}" already exists in Milvus. Comparing schema...')
                # collection already exists, compare schema with this pydantic model
                cls.__collection__ = Collection(name=collection_name, using=alias)      # type: ignore
                schema = cls.__collection__.schema      # type: ignore
                cls.__schema__ = {field.name: field for field in schema.fields}     # type: ignore
                model_fields:dict[str, MilvusField] = cls.model_fields  # type: ignore
                
                # check field correctness by comparing milvus schema with pydantic schema
                checked_fields = set()
                for field_name, field in cls.__schema__.items():    # type: ignore
                    origin_field_name = cls.ContainsField(field_name) or field_name
                    checked_fields.add(origin_field_name)
                    if field_name in cls.__non_standard_type_adapters__:
                        continue    # non-standard fields no need to check
                    try:
                        pydantic_field = model_fields[origin_field_name]
                    except KeyError:
                        raise Exception(f'Field "{field_name}" not found in model {cls.__name__}, but it found in milvus collection')
                    if not isinstance(pydantic_field, MilvusField):
                        pydantic_field = MilvusField.CastField(pydantic_field, field_name, model_cls=cls)
                    else:
                        if not pydantic_field.name:
                            pydantic_field.name = field_name
                    cls.CheckSchema(pydantic_field, field)
                unknown_fields = set(model_fields.keys()) - checked_fields  # type: ignore
                for unknown_field_name in unknown_fields:
                    # these extra fields are not in milvus schema
                    unknown_field = model_fields[unknown_field_name]
                    if not (unknown_field.non_milvus_field or unknown_field.exclude):
                        _logger.warning(f'Field "{unknown_field_name}" is not found in milvus collection schema. It will be marked as non-milvus field.')
                        unknown_field.non_milvus_field = True
                
                # check index correctness by comparing milvus index with pydantic index
                cls.__indices__ = {index.field_name: index for index in cls.__collection__.indexes()}   # type: ignore
                for field_name, index in cls.__indices__.items():   # type: ignore
                    pd_model_field_name = cls.ContainsField(field_name)
                    if not pd_model_field_name:
                        raise Exception(f'Index "{index.index_name}" on field "{pd_model_field_name}" is not defined in model')
                    elif cls.model_fields[pd_model_field_name].index is None:   
                        # create index to align with index in milvus
                        field = cls.model_fields[pd_model_field_name]
                        _logger.warning(f'Index "{index.index_name}" on field "{pd_model_field_name}" is not defined in model. It will be created to align with index info in milvus. Creation params: {index.params}')
                        field.index = MilvusIndex.CreateIndex(field.annotation, index_params=index.params)           # type: ignore                
                    cls.CheckIndex(model_fields[pd_model_field_name].index, index)   # type: ignore
                if not hasattr(cls, '__pk_field__') or not cls.__pk_field__:
                    cls.__pk_field__ = cls._GetPkFieldName(schema)  # type: ignore
                if cls.__pk_field__ is None:
                    raise Exception('There is no primary key field. Currently Milvus not supporting collection without primary key.')

            else: # create new collection
                _logger.debug(f'({cls.ClassName}.pydantic_init_subclass) Collection "{collection_name}" not found. Creating new collection...')
                
                schema:CollectionSchema = cls.BuildMilvusCollectionSchema()
                cls.__collection__ = Collection(name=collection_name, using=alias, schema=schema)   # type: ignore
                cls.CreateIndices(cls.__collection__)
                if not hasattr(cls, '__pk_field__') or not cls.__pk_field__:
                    cls.__pk_field__ = cls._GetPkFieldName(schema)  # type: ignore
                if cls.__pk_field__ is None:
                    raise Exception('There is no primary key field. Currently Milvus not supporting collection without primary key.')
                cls.__schema__ = {field.name: field for field in schema.fields}
                cls.__indices__ = {index.field_name: index for index in cls.__collection__.indexes()}             
    
    # endregion
    
    # region load
    @classmethod
    @overload
    def Load(cls): ...
    @classmethod
    @overload
    async def Load(cls, _async=True): ...
    
    @classmethod
    def Load(cls, _async=False):
        '''
        load collection to memory. 
        If `_async`=True, return a coroutine object.
        '''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        return cls.__collection__.load(sync=not _async, _async=_async)
    
    @classmethod
    async def ALoad(cls):
        '''alias of `cls.load(_async=True)`'''
        return await cls.Load(_async=True)
    # endregion
    
    @classmethod
    def Release(cls):
        '''release collection from memory'''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        cls.__collection__.release()
    
    @classmethod
    @overload
    def Drop(cls):...
    @classmethod
    @overload
    def Drop(cls, _async: Literal[False]):...
    @classmethod
    @overload
    async def Drop(cls, _async: Literal[True]):...
    
    @classmethod
    def Drop(cls, _async: bool=False):      # type: ignore
        '''!Dangerous! drop collection'''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if _async:
            return asyncio.to_thread(cls.__collection__.drop)
        cls.__collection__.drop()
        
    @classmethod
    @overload
    def IsEmpty(cls)->bool:...
    @classmethod
    @overload
    def IsEmpty(cls, _async: Literal[False])->bool:...
    @classmethod
    @overload
    async def IsEmpty(cls, _async: Literal[True])->bool:...    
    
    @classmethod
    def IsEmpty(cls, _async: bool=False):   # type: ignore
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if _async:
            async def wrapper(cls):
                return await cls.__collection__.check_is_empty(_async=True)
            return wrapper(cls)
        else:
            return cls.__collection__.is_empty
    
    @classmethod
    @overload
    def EntityCount(cls)->int:...
    @classmethod
    @overload
    def EntityCount(cls, _async: Literal[False])->int:...
    @classmethod
    @overload
    async def EntityCount(cls, _async: Literal[True])->int:...
    
    @classmethod
    def EntityCount(cls, _async: bool=False):     # type: ignore
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        return cls.__collection__.get_num_entities(_async=_async)
    
    @classmethod
    @overload
    def Partitions(cls)->list[Partition]:...
    @classmethod
    @overload
    def Partitions(cls, _async: Literal[False])->list[Partition]:...
    @classmethod
    @overload
    async def Partitions(cls, _async:Literal[True])->list[Partition]:...
    
    @classmethod
    def Partitions(cls, _async: bool=False):    # type: ignore 
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        return cls.__collection__.partitions(_async=_async)  
    
    @classmethod
    def _BuildExpr(cls, expr:str|None, **kw_exprs):
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not expr and len(kw_exprs) == 0:
            raise Exception('You must specify expr or kw_exprs')
        if expr and len(kw_exprs) != 0:
            raise Exception('You cannot use both expr and kw_exprs')
        if not expr:
            if not kw_exprs:
                raise Exception('You must specify expr or kw_exprs')
            for k in kw_exprs:
                assert k in cls.__schema__, f'Field "{k}" not found in collection'
            expr = " && ".join([f'{k}=="{v}"' for k, v in kw_exprs.items()])
        elif isinstance(expr, QueryExpression):
            expr = str(expr)
        
        if not expr:
            raise Exception('You must specify expr or kw_exprs')
        
        return expr
    
    @class_property
    def FloatVectorIndexCount(cls):
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not hasattr(cls, '_float_vector_index_count'):
            count = 0
            for field_name, field in cls.__schema__.items():
                if field.dtype in (Milvus_DataType.FLOAT_VECTOR,):
                    if field_name in cls.__indices__:
                        count += 1
            cls._float_vector_index_count = count
        return cls._float_vector_index_count
    
    @class_property
    def VectorIndexFields(cls)->list[str]:
        '''The vector field(s) of this class.'''
        if '__VectorIndexFields__' not in cls.__dict__:
            v_fields = [] 
            for field_name, field in cls.__schema__.items():
                if field_name in cls.__indices__:
                    if field.dtype in (Milvus_DataType.FLOAT_VECTOR,): 
                        v_fields.append(field_name)
            cls.__VectorIndexFields__ = v_fields    
        return cls.__VectorIndexFields__    
    
    @class_property
    def IsAutoPrimaryKey(cls)-> bool:
        '''whether the class has auto primary key'''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        return cls.model_fields[cls.__pk_field__].auto_id
    
    @class_property
    def RealPrimaryFieldName(cls)->str:
        '''return the real primary key field name. This only different to the model field name when
        you have defined the pk field's name manually in `field(...)`'''
        return cls.model_fields[cls.__pk_field__].name
    
    @class_property
    def PrimaryKeyType(cls)->type:
        '''return the type of primary key'''
        t, _, _ = _parse_anno(cls.model_fields[cls.__pk_field__].annotation)
        return t
    
    # region find
    @classmethod
    @overload
    def Find(cls, 
             expr:str|QueryExpression|None=None, 
             partition_names: Optional[list[str]] = None, 
             timeout: Optional[float] = None,
             limit: Optional[int]=None,
             offset: Optional[int]=None,
             output_fields: Optional[list[str]] = None,
             **kw_exprs)->list[Self]:...
    @classmethod
    @overload
    async def Find(cls, 
             expr:str|QueryExpression|None=None, 
             partition_names: Optional[list[str]] = None, 
             timeout: Optional[float] = None,
             limit: Optional[int]=None,
             offset: Optional[int]=None,
             _async=True,
             output_fields: Optional[list[str]] = None,
             **kw_exprs)->list[Self]:...
    
    @classmethod
    def Find(cls,       # type: ignore
             expr:str|QueryExpression|None=None, 
             partition_names: Optional[list[str]] = None, 
             timeout: Optional[float] = None,
             limit: Optional[int]=None,
             offset: Optional[int]=None,
             _async=False,
             output_fields: Optional[list[str]] = None,
             **kw_exprs):
        '''
        Find entities by expression, e.g. x=1.
        For vector similarity searching, please use "search" method.
        Args:
            expr: expression in str format
            kw_exprs: expressions in kwargs format
            partition_names: names of partitions to search
            timeout: timeout in seconds
            limit: max number of entities to return. Default is 128
            offset: offset of entities to return
            output_fields: the fields you want to retrieve. **THIS RETURNS A DICT INSTEAD OF AN OBJECT!!!!**
        
        You could type expressions in str, QueryExpression, or kwargs format.
        
        Example 1:
        ```python
        find(x=1, y=2)  # kwargs format, conditions are connected by "and"
        find('x=1 and y=2') # str format
        find(ThisModel.x == 1)  # QueryExpression
        ```
        
        Example 2:
        ```python
        # --- str format ---
        rets = await LawSummarizedFullContent.AFind(expr='chapID=="57" and scheduleID=="" and partID=="3" and sectionID=="15AA" and subsectionID=="5"')
        
        # --- query expression format ---
        # `( )` must be added between `&` since `&` has higher priority than `==`
        # note: not `and`/`or`, but `&`/`|`
        rets = await LawSummarizedFullContent.AFind(
            (LawSummarizedFullContent.chapID=="57") & \
                (LawSummarizedFullContent.scheduleID=="") & \
                    (LawSummarizedFullContent.partID=="3") & \
                        (LawSummarizedFullContent.sectionID=="15AA") & \
                            (LawSummarizedFullContent.subsectionID=="5"))
        ```
        
        For more expression format, refer to: https://milvus.io/docs/v2.3.x/boolean.md
        '''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not cls.IsLoaded:
            raise Exception(f'Collection {cls.__collection__.name} is not loaded. Please load it from attu.')
        if not limit:
            limit = 128
        expr = expr or ""
        expr = cls._BuildExpr(expr, **kw_exprs)
        if _async:
            async def wrapper(cls, expr, partition_names, timeout, limit, offset):
                result = await cls.__collection__.query(expr=expr, 
                                                        output_fields=list(cls.__schema__.keys()) if output_fields == None else output_fields,
                                                        partition_names=partition_names, 
                                                        timeout=timeout, 
                                                        limit=limit, 
                                                        offset=offset,
                                                        _async=True)
                if output_fields != None:
                    return result
                return [cls.model_validate(entity) for entity in result]
            return wrapper(cls, expr, partition_names, timeout, limit, offset)
        else:
            result = cls.__collection__.query(expr=expr, 
                                              output_fields=list(cls.__schema__.keys()) if output_fields == None else output_fields,
                                              partition_names=partition_names, 
                                              timeout=timeout, 
                                              limit=limit, 
                                              offset=offset,
                                              _async=False)
            if output_fields != None:
                return result
            return [cls.model_validate(entity) for entity in result]
    
    @classmethod
    @overload
    async def AFind(cls, 
                    expr:str|QueryExpression|None=None, 
                    partition_names: Optional[list[str]] = None, 
                    timeout: Optional[float] = None,
                    limit: Optional[int]=None,
                    offset: Optional[int]=None,
                    output_fields: Optional[list[str]] = None,
                    **kw_exprs)->list[Self]:...
    @classmethod
    @overload
    async def AFind(cls, 
                    expr:str|QueryExpression|None=None, 
                    partition_names: Optional[list[str]] = None, 
                    timeout: Optional[float] = None,
                    limit: Optional[int]=None,
                    offset: Optional[int]=None,
                    output_fields: Optional[list[str]] = None,
                    **kw_exprs)->list[Self]:...
    @classmethod
    async def AFind(cls,
                    expr:str|QueryExpression|None=None,
                    partition_names: Optional[list[str]] = None,
                    timeout: Optional[float] = None,
                    limit: Optional[int]=None,
                    offset: Optional[int]=None,
                    output_fields: Optional[list[str]] = None,
                    **kw_exprs):
        '''
        Alias of `cls.Find(_async=True)`, for making type hinting more clear.
        For vector similarity searching, please use "search" method.
        
        Args:
            expr: expression in str format
            kw_exprs: expressions in kwargs format
            partition_names: names of partitions to search
            timeout: timeout in seconds
            limit: max number of entities to return. Default is 128
            offset: offset of entities to return
            output_fields: the fields you want to retrieve. **THIS RETURNS A DICT INSTEAD OF AN OBJECT!!!!**
        
        You could type expressions in str, QueryExpression, or kwargs format.
        
        Example 1:
        ```python
        find(x=1, y=2)  # kwargs format, conditions are connected by "and"
        find('x=1 and y=2') # str format
        find(ThisModel.x == 1)  # QueryExpression
        ```
        
        Example 2:
        ```python
        # --- str format ---
        rets = await LawSummarizedFullContent.AFind(expr='chapID=="57" and scheduleID=="" and partID=="3" and sectionID=="15AA" and subsectionID=="5"')
        
        # --- query expression format ---
        # `( )` must be added between `&` since `&` has higher priority than `==`
        # note: not `and`/`or`, but `&`/`|`
        rets = await LawSummarizedFullContent.AFind(
            (LawSummarizedFullContent.chapID=="57") & \
                (LawSummarizedFullContent.scheduleID=="") & \
                    (LawSummarizedFullContent.partID=="3") & \
                        (LawSummarizedFullContent.sectionID=="15AA") & \
                            (LawSummarizedFullContent.subsectionID=="5"))
        ```
        
        For more expression format, refer to: https://milvus.io/docs/v2.3.x/boolean.md
        '''
        return await cls.Find(expr, partition_names, timeout, 
                              limit, offset, _async=True, output_fields=output_fields, **kw_exprs)      # type: ignore
        
    
    @classmethod
    def FindOne(cls, 
                expr:str|QueryExpression|None=None, 
                partition_names: Optional[list[str]] = None, 
                timeout: Optional[float] = None,
                offset: Optional[int]=None,
                output_fields: Optional[list[str]] = None,
                **kw_exprs)->Self|None:
        '''
        Find first matched entity by expression, e.g. x=1.
        For vector similarity searching, please use "search" method.
        Args:
            expr: expression in str format
            kw_exprs: expressions in kwargs format
            partition_names: names of partitions to search
            timeout: timeout in seconds
            offset: offset of entities to return
            output_fields: the fields you want to retrieve. **THIS RETURNS A DICT INSTEAD OF AN OBJECT!!!!**
        
        You could type expressions in str, QueryExpression, or kwargs format.
        
        Example 1:
        ```python
        find(x=1, y=2)  # kwargs format, conditions are connected by "and"
        find('x=1 and y=2') # str format
        find(ThisModel.x == 1)  # QueryExpression
        ```
        
        Example 2:
        ```python
        # --- str format ---
        rets = await LawSummarizedFullContent.AFind(expr='chapID=="57" and scheduleID=="" and partID=="3" and sectionID=="15AA" and subsectionID=="5"')
        
        # --- query expression format ---
        # `( )` must be added between `&` since `&` has higher priority than `==`
        # note: not `and`/`or`, but `&`/`|`
        rets = await LawSummarizedFullContent.AFind(
            (LawSummarizedFullContent.chapID=="57") & \
                (LawSummarizedFullContent.scheduleID=="") & \
                    (LawSummarizedFullContent.partID=="3") & \
                        (LawSummarizedFullContent.sectionID=="15AA") & \
                            (LawSummarizedFullContent.subsectionID=="5"))
        ```
        
        For more expression format, refer to: https://milvus.io/docs/v2.3.x/boolean.md
        '''
        kw_exprs.pop('_async', None)    # must be non-async
        matches = cls.Find(
            expr=expr, 
            partition_names=partition_names, 
            timeout=timeout, 
            limit=1, 
            offset=offset, 
            output_fields=output_fields, 
            **kw_exprs
        )
        if not matches:
            return None
        return matches[0]
    
    @classmethod
    async def AFindOne(cls, 
                expr:str|QueryExpression|None=None, 
                partition_names: Optional[list[str]] = None, 
                timeout: Optional[float] = None,
                offset: Optional[int]=None,
                output_fields: Optional[list[str]] = None,
                **kw_exprs)->Self|None:
        '''
        ASync version of `FindOne`.
        Find first matched entity by expression, e.g. x=1.
        For vector similarity searching, please use "search" method.
        Args:
            expr: expression in str format
            kw_exprs: expressions in kwargs format
            partition_names: names of partitions to search
            timeout: timeout in seconds
            offset: offset of entities to return
            output_fields: the fields you want to retrieve. **THIS RETURNS A DICT INSTEAD OF AN OBJECT!!!!**
        
        You could type expressions in str, QueryExpression, or kwargs format.
        
        Example 1:
        ```python
        find(x=1, y=2)  # kwargs format, conditions are connected by "and"
        find('x=1 and y=2') # str format
        find(ThisModel.x == 1)  # QueryExpression
        ```
        
        Example 2:
        ```python
        # --- str format ---
        rets = await LawSummarizedFullContent.AFind(expr='chapID=="57" and scheduleID=="" and partID=="3" and sectionID=="15AA" and subsectionID=="5"')
        
        # --- query expression format ---
        # `( )` must be added between `&` since `&` has higher priority than `==`
        # note: not `and`/`or`, but `&`/`|`
        rets = await LawSummarizedFullContent.AFind(
            (LawSummarizedFullContent.chapID=="57") & \
                (LawSummarizedFullContent.scheduleID=="") & \
                    (LawSummarizedFullContent.partID=="3") & \
                        (LawSummarizedFullContent.sectionID=="15AA") & \
                            (LawSummarizedFullContent.subsectionID=="5"))
        ```
        
        For more expression format, refer to: https://milvus.io/docs/v2.3.x/boolean.md
        '''
        kw_exprs.pop('_async', None)
        matches = await cls.Find(
            expr=expr, 
            partition_names=partition_names, 
            timeout=timeout, 
            limit=1, 
            offset=offset, 
            output_fields=output_fields, 
            _async=True,
            **kw_exprs
        )   # type: ignore
        if not matches:
            return None
        return matches[0]
    # endregion 
    
    # region search
    @classmethod
    def _SearchReturningFieldNames(cls)->list[str]:
        if (fs:=getattr(cls, '__default_search_out_fields__', None)) is None:
            fields = []
            for f in cls.model_fields.values():  
                if getattr(f, 'non_milvus_field', False) or f.exclude:
                    continue
                fields.append(f.name)
            cls.__default_search_out_fields__ = fields    
        return cls.__default_search_out_fields__      
    
    @classmethod
    @overload
    def Search(cls,
               data: Sequence[float] | np.ndarray | str, 
               *,
               limit: int=24,
               vector_field_name: str|None = None,
               expr:str|QueryExpression|None=None,
               partition_names: Optional[list[str]] = None, 
               timeout: Optional[float] = None,
               offset: Optional[int]=None,
               round_decimal: int = -1,
               index_params: dict[str, Any]|None=None,
               extra_fields: Sequence[str]|None = None
               )->list[list[Self]]|list[Self]:...
    @classmethod
    @overload
    def Search(cls, 
               data: Sequence[Sequence[float]] | np.ndarray | Sequence[str], 
               *,
               limit: int=24,
               vector_field_name: str|None = None,
               expr:str|QueryExpression|None=None,
               partition_names: Optional[list[str]] = None, 
               timeout: Optional[float] = None,
               offset: Optional[int]=None,
               round_decimal: int = -1,
               index_params: dict[str, Any]|None=None,
               extra_fields: Sequence[str]|None = None
               )->list[list[Self]]:...
    @classmethod
    @overload
    async def Search(cls, 
                     data: Sequence[float] | np.ndarray | str,  
                     *,
                     limit: int=24,
                     vector_field_name: str|None = None,
                     expr:str|QueryExpression|None=None,
                     partition_names: Optional[list[str]] = None, 
                     timeout: Optional[float] = None,
                     offset: Optional[int]=None,
                     round_decimal: int = -1,
                     index_params: dict[str, Any]|None=None,
                     extra_fields: Sequence[str]|None = None,
                     _async=True,
                     )->list[Self]:...
    @classmethod
    @overload
    async def Search(cls, 
                     data: Sequence[Sequence[float]] | np.ndarray | Sequence[str],  
                     *,
                     limit: int=24,
                     vector_field_name: str|None = None,
                     expr:str|QueryExpression|None=None,
                     partition_names: Optional[list[str]] = None, 
                     timeout: Optional[float] = None,
                     offset: Optional[int]=None,
                     round_decimal: int = -1,
                     index_params: dict[str, Any]|None=None,
                     extra_fields: Sequence[str]|None = None,
                     _async=True,
                     )->list[list[Self]]:...
    @classmethod
    @overload
    async def Search(cls, 
                     data: Sequence[Sequence[float]] | np.ndarray | Sequence[str],  
                     *,
                     limit: int=24,
                     vector_field_name: str|None = None,
                     expr:str|QueryExpression|None=None,
                     partition_names: Optional[list[str]] = None, 
                     timeout: Optional[float] = None,
                     offset: Optional[int]=None,
                     round_decimal: int = -1,
                     index_params: dict[str, Any]|None=None,
                     extra_fields: Sequence[str]|None = None,
                     _async=True,
                    )->list[list[Self]]:...
    
    # return scores
    @classmethod
    @overload
    def Search(cls,
               data: Sequence[float] | np.ndarray | str, 
               *,
               limit: int=24,
               vector_field_name: str|None = None,
               expr:str|QueryExpression|None=None,
               partition_names: Optional[list[str]] = None, 
               timeout: Optional[float] = None,
               offset: Optional[int]=None,
               round_decimal: int = -1,
               index_params: dict[str, Any]|None=None,
               extra_fields: Sequence[str]|None = None,
               return_scores: bool = True,
               )->list[list[tuple[Self, float]]]|list[tuple[Self, float]]:...
    @classmethod
    @overload
    def Search(cls, 
               data: Sequence[Sequence[float]] | np.ndarray | Sequence[str], 
               *,
               limit: int=24,
               vector_field_name: str|None = None,
               expr:str|QueryExpression|None=None,
               partition_names: Optional[list[str]] = None, 
               timeout: Optional[float] = None,
               offset: Optional[int]=None,
               round_decimal: int = -1,
               index_params: dict[str, Any]|None=None,
               extra_fields: Sequence[str]|None = None,
               return_scores: bool = True,
               )->list[list[tuple[Self, float]]]:...
    @classmethod
    @overload
    async def Search(cls, 
                     data: Sequence[float] | np.ndarray | str,  
                     *,
                     limit: int=24,
                     vector_field_name: str|None = None,
                     expr:str|QueryExpression|None=None,
                     partition_names: Optional[list[str]] = None, 
                     timeout: Optional[float] = None,
                     offset: Optional[int]=None,
                     round_decimal: int = -1,
                     index_params: dict[str, Any]|None=None,
                     extra_fields: Sequence[str]|None = None,
                     _async=True,
                     return_scores: bool = True,
                     )->list[tuple[Self, float]]:...
    @classmethod
    @overload
    async def Search(cls, 
                     data: Sequence[Sequence[float]] | np.ndarray | Sequence[str],  
                     *,
                     limit: int=24,
                     vector_field_name: str|None = None,
                     expr:str|QueryExpression|None=None,
                     partition_names: Optional[list[str]] = None, 
                     timeout: Optional[float] = None,
                     offset: Optional[int]=None,
                     round_decimal: int = -1,
                     index_params: dict[str, Any]|None=None,
                     extra_fields: Sequence[str]|None = None,
                     _async=True,
                     return_scores: bool = True,
                     )->list[list[tuple[Self, float]]]:...
    @classmethod
    @overload
    async def Search(cls, 
                     data: Sequence[Sequence[float]] | np.ndarray | Sequence[str],  
                     *,
                     limit: int=24,
                     vector_field_name: str|None = None,
                     expr:str|QueryExpression|None=None,
                     partition_names: Optional[list[str]] = None, 
                     timeout: Optional[float] = None,
                     offset: Optional[int]=None,
                     round_decimal: int = -1,
                     index_params: dict[str, Any]|None=None,
                     extra_fields: Sequence[str]|None = None,
                     _async=True,
                     return_scores: bool = True,
                    )->list[list[tuple[Self, float]]]:...
    
    # real implementation
    @classmethod
    def Search(cls,      #  type: ignore
               data: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | Sequence[str] | str, 
               *,
               limit: int=24,
               vector_field_name: str|None = None,
               expr:str|QueryExpression|None=None,
               partition_names: Optional[list[str]] = None, 
               timeout: Optional[float] = None,
               offset: Optional[int]=None,
               round_decimal: int = -1,
               index_params: dict[str, Any]|None=None,
               extra_fields: Sequence[str]|None = None,
               _async=False,
               return_scores: bool = False,
               ):  
        '''
        Search similar vector in specified field. Currently only support float vector.
        This is a sync function. If you want async mode, plz pass `_async=True`.
        
        Args:
            * `data`: query to search. This field can be:
                * a single vector/ list of vectors (dimension must match)
                * text / list of texts (`embedder` must be provided in field definition in this case)
            * `limit`: max number of entities to return(top n)
            * `vector_field_name`: name of vector field to search. If there are more than one float vector index, you must specify this parameter
            * `expr`: restriction expression in str format, e.g. x=1
            * `partition_names`: names of partitions to search
            * `timeout`: timeout in seconds
            * `offset`: offset of entities to return
            * `round_decimal`: round the similarity to specified decimal. If negative, no rounding
            * `index_params`: parameters for index. See pymilvus's `search` method for more details
            * `extra_fields`: extra fields to return. This is available only when `IsDynamic` is True for this model. 
                              You should fill in your target extra field keys here.
            * `_async`: whether to run this function in async mode
            * `return_scores`: whether to return the similarity score
            
        For more expression format, refer to: https://milvus.io/docs/boolean.md
        For acceptable index params(regarding to different index types), refer to: https://milvus.io/docs/index.md
        '''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not cls.IsLoaded:
            raise Exception(f'Collection {cls.__collection__.name} is not loaded. Please load it maunally from attu.')
        
        if vector_field_name is None:
            if cls.FloatVectorIndexCount != 1:
                raise Exception(f'You must specify field_name if there are more than one float vector index. Currently there are {cls.FloatVectorIndexCount} float vector indices')
            vector_field_name = next(field_name for field_name, field in cls.__schema__.items() if field.dtype in (Milvus_DataType.FLOAT_VECTOR,))
        field_info = cls.model_fields[vector_field_name]
        
        def tidy_vectors(data):
            if check_value_is(data, Sequence[float|int]):
                vectors = [data,]    
            elif check_value_is(data, np.ndarray):
                data = data.squeeze()   
                if len(data.shape) == 1:
                    assert len(data) == field_info.dimension, f'Vector dimension mismatch. Expected: {field_info.dimension}, got: {len(data)}'    
                    vectors = [data.tolist(),]
                elif len(data.shape) == 2:
                    assert data.shape[1] == field_info.dimension, f'Vector dimension mismatch. Expected: {field_info.dimension}, got: {data.shape[1]}'  
                    vectors = data.tolist()
                else:
                    raise Exception(f'Invalid shape for ndarray: {data.shape}')
            elif check_value_is(data, Sequence[Sequence[float|int]]):
                for v in data:
                    assert len(v) == field_info.dimension, f'Vector dimension mismatch. Expected: {field_info.dimension}, got: {len(v)}'                
                vectors = data
            else:
                raise Exception(f'({cls.ClassName}.Search) Invalid data type: {type(data)}')
            return vectors
        
        param={'metric_type': cls.__indices__[field_info.name].params['metric_type']}
        param.update(index_params) if index_params else None
        if _async:
            @no_type_check
            async def wrapper(cls: type[MilvusModel], data, limit, vector_field_name, expr, partition_names, timeout, round_decimal, offset):
                if isinstance(data, str):
                    data = [data,]  # will turn into vector in next `if``
                if data and isinstance(data, (tuple, list)) and isinstance(data[0], str):
                    assert field_info.embedder is not None, f'You must provide embedder for text search'    
                    vectors: list[list[float]] = await asyncio.gather(*[
                        cls.RunFieldEmbedder(field_info.embedder, d) for d in data
                    ])
                else:      
                    vectors = tidy_vectors(data)

                out_field_names = cls._SearchReturningFieldNames()  
                if cls.IsDynamic and extra_fields:
                    out_field_names = out_field_names.copy()
                    for f in extra_fields:
                        if f not in out_field_names:
                            out_field_names.append(f)
                real_vector_field_name = cls.model_fields[vector_field_name].name   
                result: list[list[Hit]] = \
                    await cls.__collection__.search(
                        data=vectors,  #  type: ignore
                        anns_field=real_vector_field_name, 
                        param=param,
                        limit=limit, 
                        expr=expr, 
                        partition_names=partition_names, 
                        output_fields=out_field_names,
                        timeout=timeout, 
                        round_decimal=round_decimal,
                        _async=True,
                        offset=offset
                    )  
                
                if len(result) == 1:
                    if return_scores:
                        return [(cls.model_validate(entity.fields), entity.distance) for entity in result[0]]
                    else:
                        return [cls.model_validate(entity.fields) for entity in result[0]]
                else:
                    if return_scores:
                        return [[(cls.model_validate(entity.fields), entity.distance) for entity in entities] for entities in result]
                    else:
                        return [[cls.model_validate(entity.fields) for entity in entities] for entities in result]
            return wrapper(cls, data, limit, vector_field_name, expr, partition_names, timeout, round_decimal, offset)
        
        else:
            if isinstance(data, str):
                data = [data,]  
            if data and isinstance(data, (tuple, list)) and isinstance(data[0], str):
                assert field_info.embedder is not None, f'You must provide embedder for text search'    
                @no_type_check
                async def _get_embeddings(embedder, data)->list[list[float]]:
                    return await asyncio.gather(*[cls.RunFieldEmbedder(embedder, d) for d in data])  
                vectors = _get_embeddings(field_info.embedder, data)   
                vectors = wait_coroutine(vectors)   
            else:
                vectors = tidy_vectors(data)
            
            out_field_names = cls._SearchReturningFieldNames()  
            if cls.IsDynamic and extra_fields:
                out_field_names = out_field_names.copy()
                for f in extra_fields:
                    if f not in out_field_names:
                        out_field_names.append(f)
            real_vector_field_name = cls.model_fields[vector_field_name].name   
            result: list[list[Hit]] = \
                cls.__collection__.search(
                    data=vectors,    # type: ignore
                    anns_field=real_vector_field_name, 
                    param=param,
                    limit=limit, 
                    expr=expr, 
                    partition_names=partition_names, 
                    output_fields=out_field_names,
                    timeout=timeout,
                    round_decimal=round_decimal,
                    _async=False,
                    offset=offset,
                )   # type: ignore
            if len(result) == 1:
                if return_scores:
                    return [(cls.model_validate(entity.fields), entity.distance) for entity in result[0]]
                else:
                    return [cls.model_validate(entity.fields) for entity in result[0]]
            else:
                if return_scores:
                    return [[(cls.model_validate(entity.fields), entity.distance) for entity in entities] for entities in result]
                else:
                    return [[cls.model_validate(entity.fields) for entity in entities] for entities in result]
    
    @classmethod
    @overload
    async def ASearch(cls,
                      data: Sequence[float] | np.ndarray | str, 
                      *,
                      limit: int=24,
                      vector_field_name: str|None = None,
                      expr:str|QueryExpression|None=None,
                      partition_names: Optional[list[str]] = None, 
                      timeout: Optional[float] = None,
                      offset: Optional[int]=None,
                      round_decimal: int = -1,
                      index_params: dict[str, Any]|None=None)->list[Self]:...
    @classmethod
    @overload
    async def ASearch(cls,
                      data: Sequence[Sequence[float]] | np.ndarray | Sequence[str], 
                      *,
                      limit: int=24,
                      vector_field_name: str|None = None,
                      expr:str|QueryExpression|None=None,
                      partition_names: Optional[list[str]] = None, 
                      timeout: Optional[float] = None,
                      offset: Optional[int]=None,
                      round_decimal: int = -1,
                      index_params: dict[str, Any]|None=None)->list[list[Self]]:...
    @classmethod
    @overload
    async def ASearch(cls,
                      data: Sequence[float] | np.ndarray | str, 
                      *,
                      limit: int=24,
                      vector_field_name: str|None = None,
                      expr:str|QueryExpression|None=None,
                      partition_names: Optional[list[str]] = None, 
                      timeout: Optional[float] = None,
                      offset: Optional[int]=None,
                      round_decimal: int = -1,
                      index_params: dict[str, Any]|None=None,
                      return_scores: bool = True,
                      )->list[tuple[Self, float]]:...
    @classmethod
    @overload
    async def ASearch(cls,
                      data: Sequence[Sequence[float]] | np.ndarray | Sequence[str], 
                      *,
                      limit: int=24,
                      vector_field_name: str|None = None,
                      expr:str|QueryExpression|None=None,
                      partition_names: Optional[list[str]] = None, 
                      timeout: Optional[float] = None,
                      offset: Optional[int]=None,
                      round_decimal: int = -1,
                      index_params: dict[str, Any]|None=None,
                      return_scores: bool = True,
                    )->list[list[tuple[Self, float]]]:...    

    @classmethod
    async def ASearch(cls,      # type: ignore
                      data: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | Sequence[str] | str, 
                      *,
                      limit: int=24,
                      vector_field_name: str|None = None,
                      expr:str|QueryExpression|None=None,
                      partition_names: Optional[list[str]] = None, 
                      timeout: Optional[float] = None,
                      offset: Optional[int]=None,
                      round_decimal: int = -1,
                      index_params: dict[str, Any]|None=None,
                      return_scores: bool = False
                    ):
        '''
        Alias of `cls.Search` method with param _async=True.
        This is for making type hinting more clear.
        
        Args:
            * `data`: query to search. This field can be:
                * a single vector/ list of vectors (dimension must match)
                * text / list of texts (`embedder` must be provided in field definition in this case)
            * `limit`: max number of entities to return(top n)
            * `vector_field_name`: name of vector field to search. If there are more than one float vector index, you must specify this parameter
            * `expr`: restriction expression in str format, e.g. x=1
            * `partition_names`: names of partitions to search
            * `timeout`: timeout in seconds
            * `offset`: offset of entities to return
            * `round_decimal`: round the similarity to specified decimal. If negative, no rounding
            * `index_params`: parameters for index. See pymilvus's `search` method for more details
            * `extra_fields`: extra fields to return. This is available only when `IsDynamic` is True for this model. 
                              You should fill in your target extra field keys here.
            * `_async`: whether to run this function in async mode
            * `return_scores`: whether to return the similarity score
            
        For more expression format, refer to: https://milvus.io/docs/boolean.md
        For acceptable index params(regarding to different index types), refer to: https://milvus.io/docs/index.md
        '''
        return await cls.Search(
            data, 
            limit=limit, 
            vector_field_name=vector_field_name, 
            expr=expr, 
            partition_names=partition_names, 
            timeout=timeout, 
            offset=offset, 
            round_decimal=round_decimal, 
            index_params=index_params, 
            _async=True,
            return_scores=return_scores,
        )   
    # endregion
    
    # region get
    @classmethod
    @overload
    def Get(cls, pk:int|str)->Self|None:...
    @classmethod
    @overload
    async def Get(cls, pk:int|str, _async=True)->Self|None:...
    
    @classmethod
    def Get(cls, pk:int|str, _async=False):
        '''
        Return an entity with the given primary key.
        If there is no entity with the given primary key, return None.
        '''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if _async:
            async def wrapper(cls):
                entity = await cls.Find(_async=True, **{cls.__pk_field__: pk}) 
                if len(entity) == 0:
                    return None
                return entity[0]
            return wrapper(cls)
        else:
            entity = cls.Find(**{cls.__pk_field__: pk})     # type: ignore
            if len(entity) == 0:    
                return None
            return entity[0]    
    
    @classmethod
    async def AGet(cls, pk:int|str):
        '''Alias of `cls.Get(_async=True)`'''
        return await cls.Get(pk, _async=True)
    # endregion
    
    # region delete
    @classmethod
    @overload
    def Delete(cls, expr: str|QueryExpression|None=None, partition_name: Optional[str] = None, timeout: Optional[float] = None, **kw_exprs)->MutationResult:...
    @classmethod
    @overload
    async def Delete(cls, expr: str|QueryExpression|None=None, partition_name: Optional[str] = None, timeout: Optional[float] = None, _async=True, **kw_exprs)->MutationResult:...
        
    @classmethod
    def Delete(cls, expr: str|QueryExpression|None=None, partition_name: Optional[str] = None, timeout: Optional[float] = None, _async=False, **kw_exprs):  # type: ignore  
        '''
        Delete entities by expression, e.g. x=1.
        Note that expr could only be `None` or empty when `kw_exprs` is provided.
        '''
        if cls.__is_abstract__:
            raise Exception(f'Class {cls.__name__} is abstract')
        if not cls.IsLoaded:
            raise Exception(f'Collection {cls.__collection__.name} is not loaded. Please load it maunally from attu.')
        expr = cls._BuildExpr(expr, **kw_exprs)
        return cls.__collection__.delete(expr, partition_name=partition_name, timeout=timeout, _async=_async)
    
    @classmethod
    async def ADelete(cls, expr: str|QueryExpression|None=None, partition_name: Optional[str] = None, timeout: Optional[float] = None, **kw_exprs):
        '''alias of `cls.Delete(_async=True)`'''
        return await cls.Delete(expr, partition_name, timeout, _async=True, **kw_exprs)     # type: ignore
    # endregion
    
    # region save
    @classmethod
    def _ExcludeSavingFields(cls)->set[str]:
        if not hasattr(cls, '__exclude_saving_fields__'):
            model_dump_exclude_fields = set()
            model_dump_exclude_fields.update(cls.__non_standard_type_adapters__.keys())
            model_dump_exclude_fields.update(cls.__other_db_fields__.keys())
            for fn, f in cls.model_fields.items():
                if f.exclude or getattr(f, 'non_milvus_field', False):
                    model_dump_exclude_fields.add(fn)
            setattr(cls, '__exclude_saving_fields__', model_dump_exclude_fields)
        return getattr(cls, '__exclude_saving_fields__')    
    
    def dump_for_save(self)->dict[str, Any]:
        '''
        This method defines how this model should be dumped for saving
        to Milvus. You can override this method to customize the save behavior.
        The final return should be a dict.
        '''
        data = self.model_dump(exclude=self._ExcludeSavingFields())
        
        # dump non-standard types to str
        for field_name, adapter in self.__non_standard_type_adapters__.items():
            field_info = self.__class__.model_fields[field_name]
            real_field_name = field_info.name or field_name
            data[real_field_name] = adapter.dump_json(getattr(self, field_name)).decode()   # change bytes to str
        
        # dump db fields to as primary keys
        for field_name in self.__other_db_fields__:
            field_info = self.__class__.model_fields[field_name]
            real_field_name = field_info.name or field_name
            val = getattr(self, field_name)
            if val is None:
                data[real_field_name] = ""  # use empty string to represent None
            elif isinstance(val, (int, str)):
                data[real_field_name] = val
            elif isinstance(val, MilvusModel):
                data[real_field_name] = val.pk
            else:
                raise Exception(f'Invalid type for field {field_name}: {type(val)}. Cannot dump as primary key to save in Milvus')
        return data
    
    @classmethod
    @overload
    def Save(cls, items:Self|Sequence[Self], partition_name:str|None=None, timeout: Optional[float] = None,)->MutationResult:...
    @classmethod
    @overload
    async def Save(cls, items:Self|Sequence[Self], partition_name:str|None=None, timeout: Optional[float] = None, _async=True)->MutationFuture:...
    
    @classmethod
    def Save(cls,       # type: ignore
             items:Self|Sequence[Self], 
             partition_name:str|None=None, 
             timeout: Optional[float] = None, 
             _async=False):  
        '''
        Save instance(s) to Milvus. When `_async`=True, return a coroutine object.
        Note: For auto integer ID, (`auto_id`=True and pk is `int`), the real PK will be set back to the entity
              after the entity is saved. 
        '''
        if isinstance(items, MilvusModel):
            items = [items,]
        
        insert_items: list[Self] = []
        upsert_items: list[Self] = []
        
        is_int_auto_key_model = (cls.IsAutoPrimaryKey and check_type_is(cls.PrimaryKeyType, int))
        if is_int_auto_key_model:
            key_field_name = cls.__pk_field__
            for item in items:
                if getattr(item, key_field_name) == -1:
                    insert_items.append(item)   
                else:
                    upsert_items.append(item)   
        else:
            upsert_items = list(items)  
        
        if not insert_items and not upsert_items:
            return _EmptyMutationResult()
        
        def pack_data(cls, data: list[dict], ignore_pk=False):
            packed = []
            for field_name, field_info in cls.model_fields.items():
                if (field_name == cls.__pk_field__ and ignore_pk) or field_info.exclude:  
                    continue
                packed.append([d[field_info.name] for d in data])
            return packed
        
        if _async:
            async def wrapper(cls, up_items: list[Self], in_items: list[Self]):
                r_up, r_in = None, None     # type: ignore
                if up_items:
                    up_data = [item.dump_for_save() for item in up_items]
                    packed_up_data = pack_data(cls, up_data, ignore_pk=False)
                    r_up: MutationResult = await cls.__collection__.upsert(packed_up_data,
                                                                           partition_name=partition_name, 
                                                                           timeout=timeout, 
                                                                           _async=True)    
                if in_items:
                    in_data = [item.dump_for_save() for item in in_items]
                    packed_in_data = pack_data(cls, in_data, ignore_pk=True)
                    r_in: MutationResult = await cls.__collection__.insert(packed_in_data,
                                                                           partition_name=partition_name, 
                                                                           timeout=timeout, 
                                                                           _async=True)
                    for i, pk in enumerate(r_in._primary_keys):
                        setattr(in_items[i], cls.__pk_field__, pk)  
                        # set the pk value back to the entity
                        # i.e. `-1` -> `.....`(real int auto pk)
                return CombineMutationResult(r_up, r_in)        # type: ignore
                
            return wrapper(cls, upsert_items, insert_items)   

        r_up, r_in = None, None     # type: ignore
        if upsert_items:
            up_data = [item.dump_for_save() for item in upsert_items]
            packed_up_data = pack_data(cls, up_data, ignore_pk=False)
            r_up: MutationResult = cls.__collection__.upsert(packed_up_data,
                                                             partition_name=partition_name, 
                                                             timeout=timeout, 
                                                             _async=False)    
        if insert_items:
            in_data = [item.dump_for_save() for item in insert_items]
            packed_in_data = pack_data(cls, in_data, ignore_pk=True)
            r_in: MutationResult = cls.__collection__.insert(packed_in_data,
                                                             partition_name=partition_name, 
                                                             timeout=timeout, 
                                                             _async=False)
            for i, pk in enumerate(r_in._primary_keys):
                setattr(insert_items[i], cls.__pk_field__, pk)  
                # set the pk value back to the entity
                # i.e. `-1` -> `.....`(real int auto pk)
        return CombineMutationResult(r_up, r_in)    # type: ignore   
    
    @classmethod
    async def ASave(cls, items:Self|Sequence[Self], partition_name:str|None=None, timeout: Optional[float] = None)->MutationFuture:
        '''alias of `cls.Save(_async=True)`'''
        return await cls.Save(items, partition_name, timeout, _async=True)
    # endregion
    
    # region item operations
    @property
    def pk(self):
        '''return the primary key of this entity'''
        return getattr(self, self.__pk_field__)
    
    @overload
    def save(self, partition_name:str|None=None, timeout: Optional[float] = None,)->MutationResult:...
    @overload
    async def save(self, partition_name:str|None=None, timeout: Optional[float] = None, _async=True)->MutationResult:...
    
    def save(self, partition_name:str|None=None, timeout: Optional[float] = None, _async=False):        # type: ignore
        '''
        Save entity to collection. 
        It will automatically using `insert` or `upsert`, depends on the condition of this entity
        '''
        return self.__class__.Save(self, partition_name=partition_name, timeout=timeout, _async=_async)
    
    async def asave(self, partition_name:str|None=None, timeout: Optional[float] = None):
        '''alias of `self.save(_async=True)`'''
        return await self.save(partition_name, timeout, _async=True)
    
    @classmethod
    @overload
    async def RunFieldEmbedder(cls, embedder: str|MilvusEmbedderType[Self], data: str)->list[float]:...
    @classmethod
    @overload
    async def RunFieldEmbedder(cls, embedder: str|MilvusEmbedderType[Self], data: str, return_list: Literal[False]=False)->np.ndarray:...
    @classmethod
    @overload
    async def RunFieldEmbedder(cls, embedder: str|MilvusEmbedderType[Self], data: str, return_list: Literal[True]=True)->list[float]:...
    @classmethod
    @no_type_check
    async def RunFieldEmbedder(cls, embedder: str|MilvusEmbedderType[Self], data: str, return_list: bool=True): 
        '''
        Run the a embedder for the given field.
        
        Args:
            - `embedder`: the field name or the embedder function itself. It can be a string(field name) or a callable function.
            - `data`: the data to be embedded. It can be a string or a list of strings.
            - `return_list`: whether to return the result as a list of float or a numpy array.
        '''
        if isinstance(embedder, str):
            if not (field_name := cls.ContainsField(embedder)):
                raise Exception(f'Field {embedder} not found in model {cls.__name__}')
            embedder = cls.model_fields[field_name].embedder
        if not embedder:
            raise Exception(f'Field {embedder} not found in model {cls.__name__}')
        
        if func_arg_count(embedder) == 2:
            embed = embedder(cls, data)   
        else:
            embed = embedder(data)    
        if isinstance(embed, Coroutine):
            embed = await embed
        if isinstance(embed, np.ndarray):
            if return_list:
                return embed.tolist()
            return embed
        else:
            if not return_list:
                embed = np.array(embed)
            return list(embed)
    
    async def similarity(self, obj: "str|list[float]|np.ndarray|Self", compare_field: str|None=None)->float:
        '''
        Calculate the similarity between this entity and the given object(embedding).
        The object can be:
            * text (must ensure `embedder` is provided in field definition)
            * embedding (list of float/numpy array, must ensure dimension correct)
            * another entity of the same type
            
        `compare_field` can only be None when this model has only one vector field.
        '''
        if not compare_field:
            if len(self.VectorIndexFields)>1:
                raise Exception(f'You must specify compare_field if there are more than one vector field')
            compare_field = self.VectorIndexFields[0]
        else:
            _compare_field = self.ContainsField(compare_field)
            if not _compare_field:
                raise Exception(f'Field {compare_field} not found in model {self.__class__.__name__}')
            if _compare_field not in self.VectorIndexFields:
                raise Exception(f'Field {compare_field} is not a vector field')
            compare_field = _compare_field
            
        if isinstance(obj, str):
            embedding = await self.RunFieldEmbedder(compare_field, obj, return_list=False)
        elif check_value_is(obj, Sequence[float|int]):
            embedding = np.array(obj)
        elif isinstance(obj, MilvusModel):
            embedding = np.array(getattr(obj, compare_field))
        elif not isinstance(obj, np.ndarray):
            raise Exception(f'Invalid type for object: {type(obj)}. Must be str or list of float')
        return _cosine_similarity(getattr(self, compare_field), embedding)   
    # endregion

class MilvusParentChildModel(MilvusModel, abstract=True):
    '''
    Abstract milvus model protocol for parent-child liked data structure.
    Subclasses must define `parent`/`children` methods.
    Some other optional methods can also be overwritten for better performance.
    
    Model class args:
        db (str|Team): The db name of the model. It can be string or team(team name will be used in this case).
                       Default to be `default`, which is a common area for all teams. 
                       Note that if db==`global`, it will be directed to `default`.
        collection_name (str|None): the collection name of the model. If not specified, the class name will be used.
                                    Default to be None.
        alias:(str|None): The alias of the collection. If not specified, the db name + collection name will be used.
                            Default to be None.
        abstract(bool): In case you want to be an abstract base class to provide some common fields,
                        set this to True. Default to be False. Abstract model will not be created to Milvus.
    
    Note: You could assign collection description by typing comment under model class
    '''
    
    @abstractmethod
    async def parent(self)->Self|None:
        '''
        Return the parent of this entity.
        If there is no parent(root node), return None.
        '''
    
    @abstractmethod
    async def children(self)->list[Self]:
        '''Return all direct children of this entity.
        Grand children should not included.'''
        
    async def siblings(self)->list[Self]:
        '''Return all siblings of this entity(exclude itself)'''
        par = await self.parent()
        if not par:
            return []
        children = await par.children()
        return [child for child in children if child != self]

    async def distance(self, other: Self)->int|None:
        '''
        Return the height distance difference between this entity
        and the other entity.
        Returns:
            - zero: `other` has same parent as this node(i.e. sibling)
            - positive: when `other` is a child of this node
            - negative: when `other` is a parent of this node
            - none: when 2 entities are staying in different trees.
        You can override this method to do your own implementation.
        '''
        if other == self:
            return 0
        
        curr, h = self, 0
        while curr:
            h += 1
            curr = await self.parent()
            if curr == other:
                return -h
        curr, h = other, 0
        while curr:
            h += 1
            curr = await other.parent()
            if curr == self:
                return h
        return None  
    
    async def is_sibling(self, other: Self)->bool:
        '''
        Check if this entity is a sibling of `other`.
        You can override this method to implement your own algorithm.
        '''
        if self == other:
            return True
        self_par = await self.parent()
        other_par = await other.parent()
        return self_par == other_par
    
    async def height(self)->int:
        '''
        Return the height of this entity in the tree.
        Root node(no parent) should return 0.
        You can override this method to do your own implementation
        for better performance.
        '''
        if not '__height__' in self.__dict__:
            height = 0
            par = await self.parent()
            while par:
                height += 1
                par = await par.parent()
            self.__height__ = height    
        return self.__height__  
    
    async def is_child(self, other: Self)->bool:
        '''
        Check if this entity is a child of `other` (including grand children).
        You can override this method to implement your own algorithm.
        '''
        if other.pk == self.pk:
            return False
        return await other.is_parent(self)

    async def is_parent(self, other: Self)->bool:
        '''
        Check if this entity is a parent of `other`.
        You can override this method to implement your own algorithm.
        '''
        if other.pk == self.pk:
            return False
        other_parent = await other.parent()
        while other_parent:
            if other_parent.pk == self.pk:
                return True
            other_parent = await other_parent.parent()
        return False

    async def common_parent(self, other: Self)->Self|None:
        '''
        Find the common parent of this entity and the other entity.
        Return none if there is no common parent.
        '''
        if other == self:
            return self
        other_level = await other.height()
        self_level = await self.height()
        s, o = self, other
        
        async def go_up(n: Self, h: int)->Self|None:
            curr = n
            while curr and h>0:
                curr = await curr.parent()
                h -= 1
            return curr
        
        if other_level < self_level:
            s = await go_up(s, self_level - other_level)
            if not s:
                return None
        elif other_level > self_level:
            o = await go_up(o, other_level - self_level)
            if not o:
                return None
        if s == o:
            return s
        while s and o:
            s = await s.parent()
            o = await o.parent()
            if s == o:
                return s
        return None
    


__all__ = ['MilvusConnectionConfig', 'MilvusModel', 'MilvusParentChildModel']



if __name__ == '__main__':
    class Test2(MilvusModel, db='test_db'):
        x: int = field(is_pk=True, auto_id=True)    # default to be `-1`
        y: list[float] = field(is_vector=True, dimension=8)
        
    t1 = Test2(y=[1,1,1,1,1,1,1,1])
    t2 = Test2(y=[1,1,1,1,1,1,1,1])
    
    print(t1.x, t2.x)   # before: -1, -1
    r = Test2.Save([t1, t2])
    print(t1.x, t2.x)   # after(real pks): 'xxxxx', 'xxx...'
    