# -*- coding: utf-8 -*-
'''base types for Milvus ORM'''

if __name__ == "__main__": # for debugging
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'thinkthinksyn.db_utils.milvus'

import numpy as np

from abc import ABC, abstractmethod
from types import UnionType, NoneType
from typing import Literal, TYPE_CHECKING, ClassVar, Union, Annotated, TypeAlias
from typing_extensions import TypeForm

from ...common_utils.type_utils import (check_value_is, get_sub_clses, get_origin, get_args)

if TYPE_CHECKING:
    from .fields import MilvusField

def _fuzzy_simply(text: str)-> str:
    return text.lower().replace(' ', '').replace('_', '').replace('-', '')

def _fuzzy_compare(a: str, b: str)-> bool:
    return _fuzzy_simply(a) == _fuzzy_simply(b)

def _parse_anno(t)->tuple[type, tuple, bool]:
    '''
    Returns:
        1. type of the annotation, e.g. optional[int] -> int
        2. args of the annotation, e.g. List[int] -> (int,)
        3. whether the anno is optional 
    '''
    origin = get_origin(t, return_t_if_no_origin=True)
    args = get_args(t)
        
    if origin in (UnionType, Union) and len(args)==2:
        for i, a in enumerate(args):
            if a in (NoneType, None):
                return args[1-i], tuple(), True
    elif origin == Annotated:
        return _parse_anno(args[0])
    return origin, args, False  # type: ignore
    
# region index base
class MilvusIndex(ABC):
    '''
    ABC of milvus index. Do not use this class directly. Use sub class instead
    
    Subclass Args:
        * index_type: str|None: type of index. If None, this is a sub-abstract class
    '''
    index_type: ClassVar[str|None] = None
    
    def __init__(self, name:str|None=None):
        self.name = name
        
    def __init_subclass__(cls, index_type: str|None=None) -> None:
        cls.index_type = index_type
        
    @staticmethod
    def CreateIndex(field_anno: type|TypeForm, index_type: str|None=None, index_params: dict={})->"MilvusIndex":
        '''create index by looking to the fields' annotation type, with the params provided.'''
        if not index_type and 'index_type' in index_params:
            index_type = index_params['index_type']
            
        origin, args, is_optional = _parse_anno(field_anno)
        if issubclass(origin, (list, tuple, np.ndarray)):
            inner = args[0] if args else None
            if not inner or issubclass(inner, float):
                index_cls = FloatVectorIndex
            else:
                raise Exception(f'Unsupported field type for index: {origin}')    
        elif issubclass(origin, str):
            index_cls = StringIndex
        else:
            raise Exception(f'Unsupported field type for index: {origin}')
        
        if index_type:
            for sub_cls in get_sub_clses(index_cls):
                if sub_cls.index_type:
                    if _fuzzy_compare(sub_cls.index_type, index_type):
                        index_cls = sub_cls
                        break
            else:
                raise Exception(f'Index type {index_type} is not supported for field type {origin}')
        index_params.pop('index_type', None)
        return index_cls(**index_params)
        
    @abstractmethod 
    def validate_field(self, field:'MilvusField'):
        '''
        Override this method to specify the validation of this index on this field.
        No return. Just raise exception if validation failed.
        '''
        pass
    
    @abstractmethod
    def validate_index(self, index_params: dict):
        if index_params['index_type'] != self.index_type:
            raise Exception(f'Index type of string index must be {self.index_type}')
        
    @abstractmethod
    def build_create_index_params(self, field_name:str)->dict:
        '''build params for building milvus index'''
        pass

class VectorIndex(MilvusIndex):
    '''base abstract class for vector index'''

class ScalarIndex(MilvusIndex):
    '''base abstract class for scalar index'''
# endregion

# region string index
StringIndexType = Literal['MARISA_TRIE']
class StringIndex(ScalarIndex):
    '''Index for VARCHAR. Currently Milvus only support "MARISA_TRIE" index'''
    
    index_type: StringIndexType
    
    def __init_subclass__(cls, index_type: str|None = None) -> None:
        if not check_value_is(index_type, StringIndexType): # type: ignore
            raise Exception(f'Index type of string field must be one of {get_args(StringIndexType)}')
        super().__init_subclass__(index_type)
    
    def __init__(self, name: str|None = None):
        '''Currently milvus only support "MARISA_TRIE" index'''
        super().__init__(name)
    
    def validate_field(self, field: 'MilvusField'):
        field_type, field_args, is_optional = _parse_anno(field.annotation)
        if not issubclass(field_type, str):
            raise Exception('String index can only be applied on string field')
    
    def validate_index(self, index_params: dict):
        if 'params' not in index_params:
            if not len(index_params) == 0:
                raise Exception('String index does not need any params, but got some on milvus side')
        else:
            if not len(index_params['params']) == 0:
                raise Exception('String index does not need any params, but got some on milvus side')
    
    def build_create_index_params(self, field_name: str) -> dict:
        index_name = self.name if self.name is not None else f'{field_name}_index'
        return {
            'field_name': field_name,
            'index_name': index_name,
            # currently milvus(2.3.5) doesn't need VARCHAR index to provide index type(since it only support one.)
        }
        
class StringIndex_MARISA_TRIE(StringIndex, index_type='MARISA_TRIE'):
    '''Currently, Milvus only support "MARISA_TRIE" index for string fields'''

NumericIndexType: TypeAlias = Literal['STL_SORT']

class NumericIndex(ScalarIndex):
    """
    Indexing with numeric scalar fields.
    Doc: https://milvus.io/docs/index-scalar-fields.md 
    """

    index_type: NumericIndexType
    
    def __init_subclass__(cls, index_type: NumericIndexType|None=None) -> None:
        if index_type is None:
            return
        if not check_value_is(index_type, NumericIndexType):    # type: ignore
            raise Exception(f'Index type of numeric field must be one of {get_args(NumericIndexType)}')
        super().__init_subclass__(index_type)
    
    def __init__(self, name: str|None = None):
        print('init numeric index')
        super().__init__(name)
    
    def validate_field(self, field: 'MilvusField'):
        field_type, field_args, is_optional = _parse_anno(field.annotation)
        if not issubclass(field_type, (int, float)):
            raise Exception('Numeric index can only be applied on numeric field')
    
    def validate_index(self, index_params: dict):
        return super().validate_index(index_params)
    
    def build_create_index_params(self, field_name: str) -> dict:
        index_name = self.name if self.name is not None else f'{field_name}_index'
        return {
            'field_name': field_name,
            'index_name': index_name,
            'index_type': self.index_type,
        }

class NumericIndex_STL_SORT(NumericIndex, index_type='STL_SORT'):
    pass

# endregion

# region float vector index
FloatVectorIndexType = Literal['FLAT', 'IVF_FLAT', 'GPU_IVF_FLAT', 'IVF_SQ8', 'IVF_PQ', 'GPU_IVF_PQ', 'HNSW', 'SCANN', 'ANNOY', 'AUTOINDEX', 'DISKANN']
FloatVectorMetricType = Literal['L2', 'IP', 'COSINE']
class FloatVectorIndex(VectorIndex):
    '''index for float vector'''
    
    index_type: FloatVectorIndexType
    '''type of index, e.g. 'FLAT', 'IVF_FLAT', 'GPU_IVF_FLAT', 'IVF_SQ8', 'DISKANN'... '''
    
    def __init_subclass__(cls, index_type: FloatVectorIndexType|None=None) -> None:
        if index_type is None:
            return # for sub-abstract class
        elif not check_value_is(index_type, FloatVectorIndexType):
            raise Exception(f'Index type of float vector field must be one of {get_args(FloatVectorIndexType)}')
        super().__init_subclass__(index_type)
    
    def __init__(self, name: str|None = None, metric_type: FloatVectorMetricType = 'IP'):
        super().__init__(name)
        if not check_value_is(metric_type, FloatVectorMetricType):
            raise Exception(f'Metric type of float vector field must be one of {get_args(FloatVectorMetricType)}')
        self.metric_type = metric_type
        
    def validate_field(self, field: 'MilvusField'):
        field_type, field_args, is_optional = _parse_anno(field.annotation)
        if not issubclass(field_type, (list, tuple, np.ndarray)):
            raise Exception('Float vector index can only be applied on float vector field')
        sub_type = field_args[0] if field_args else float
        if not (issubclass(sub_type, float) or sub_type in (np.float64, np.float32, np.float16)):
            raise Exception('Float vector index can only be applied on float vector field')
        
    def validate_index(self, index_params: dict):
        super().validate_index(index_params)
        if 'metric_type' not in index_params:
            raise Exception('metric_type must be provided for float vector index')
        if index_params['metric_type'] != self.metric_type:
            raise Exception(f'metric_type mismatch. Expect {self.metric_type}, but got {index_params["metric_type"]}')
        
    def build_create_index_params(self, field_name: str) -> dict:
        index_name = self.name if self.name is not None else f'{field_name}_index'
        if not self.name:
            self.name = index_name  # set back to self.name for future reference
        return {
            'field_name': field_name,
            'index_name': index_name,
            'index_type': self.index_type,
            'metric_type': self.metric_type,
            'params': {}
        }

class _NListFloatVectorIndex(FloatVectorIndex):
    
    def __init__(self, name: str|None = None, metric_type: FloatVectorMetricType = 'IP', nlist:int=128):
        '''
        Args:
            name: name of index. If not set, it will be `field_name` + '_index'
            metric_type: metric type of index. Must be one of 'L2', 'IP', 'COSINE'
            nlist: number of "clusters". More clusters means more accurate but slower
        '''
        super().__init__(name, metric_type=metric_type)
        if not nlist in range(1, 65536):
            raise Exception('nlist must be in range [1, 65536]')
        self.nlist = nlist
        
    def validate_index(self, index_params: dict):
        super().validate_index(index_params)
        if 'params' not in index_params:
            raise Exception('params must be provided for float vector index')
        if 'nlist' not in index_params['params']:
            raise Exception('"nlist" does not occur in Milvus index. Index setting mismatch') 
        
    def build_create_index_params(self, field_name: str) -> dict:
        params = super().build_create_index_params(field_name)
        params['params'] = {
            'nlist': self.nlist,
        }
        return params

class FloatVectorIndex_FLAT(FloatVectorIndex, index_type='FLAT'):
    def __init__(self, name: str|None = None, metric_type: FloatVectorMetricType = 'IP'):
        super().__init__(name, metric_type=metric_type)

class FloatVectorIndex_IVF_FLAT(_NListFloatVectorIndex, index_type='IVF_FLAT'):...

class FloatVectorIndex_GPU_IVF_FLAT(_NListFloatVectorIndex, index_type='GPU_IVF_FLAT'):...

class FloatVectorIndex_IVF_SQ8(_NListFloatVectorIndex, index_type='IVF_SQ8'):...

class FloatVectorIndex_IVF_PQ(_NListFloatVectorIndex, index_type='IVF_PQ'):
    
    def __init__(self, 
                 name: str|None = None, 
                 metric_type: FloatVectorMetricType = 'IP', 
                 nlist:int=128, 
                 m:Literal[4,8,16,32,64]=4, 
                 nbits: int=8):
        '''
        Args:
            nlist: number of "clusters". More clusters means more accurate but slower
            m: number of factors of product quantization.
            nbits: number of bits of each sub vector. Must be in range [1, 16]
        '''
        super().__init__(name, metric_type=metric_type, nlist=nlist)
        if not nbits in range(1, 17):
            raise Exception('nbits must be in range [1, 16]')
        if not m in (4,8,16,32,64):
            raise Exception('m must be one of (4,8,16,32,64)')
        self.nbits = nbits
        self.m = m
        
    def build_create_index_params(self, field_name: str) -> dict:
        params = super().build_create_index_params(field_name)
        params['params']['m'] = self.m  # type: ignore
        params['params']['nbits'] = self.nbits  # type: ignore
        return params

class FloatVectorIndex_GPU_IVF_PQ(_NListFloatVectorIndex, index_type='GPU_IVF_PQ'):...

class FloatVectorIndex_HNSW(FloatVectorIndex, index_type='HNSW'):
    def __init__(self, 
                 name: str|None = None, 
                 metric_type: FloatVectorMetricType = 'IP', 
                 m:int = 16,
                 efConstruction:int=256):
        '''
        @param m: Maximum degree of the node, range[4, 64]
        @param efConstruction: number of neighbors to explore at construction time. Must be in range [8, 512]
        '''
        super().__init__(name, metric_type=metric_type)
        if not m in range(4, 65):
            raise Exception('m must be in range [4, 64]')
        if not efConstruction in range(8, 513):
            raise Exception('efConstruction must be in range [8, 512]')
        self.m = m
        self.efConstruction = efConstruction
    def validate_index(self, index_params: dict):
        super().validate_index(index_params)
        if 'params' not in index_params:
            raise Exception('params must be provided for float vector index')
        if 'M' not in index_params['params']:
            raise Exception('"M" does not occur in Milvus index. Index setting mismatch')
        if 'efConstruction' not in index_params['params']:
            raise Exception('"efConstruction" does not occur in Milvus index. Index setting mismatch')
    def build_create_index_params(self, field_name: str) -> dict:
        params = super().build_create_index_params(field_name)
        params['params'] = {
            'M': self.m,
            'efConstruction': self.efConstruction,
        }
        return params
    
class FloatVectorIndex_AUTOINDEX(FloatVectorIndex, index_type='AUTOINDEX'):...
    
class FloatVectorIndex_SCANN(_NListFloatVectorIndex, index_type='SCANN'):...
    
class FloatVectorIndex_ANNOY(FloatVectorIndex, index_type='ANNOY'):
    def __init__(self, 
                 name: str|None = None, 
                 metric_type: FloatVectorMetricType = 'IP', 
                 n_trees:int=10):
        '''
        @param n_trees: number of trees in index. Must be in range [1, 1024]
        '''
        super().__init__(name, metric_type=metric_type)
        if not n_trees in range(1, 1025):
            raise Exception('n_trees must be in range [1, 1024]')
        self.n_trees = n_trees
        
    def validate_index(self, index_params: dict):
        super().validate_index(index_params)
        if 'params' not in index_params:
            raise Exception('In milvus, this index does not have params, while in ORM it has')
        if 'n_trees' not in index_params['params']:
            raise Exception('"n_trees" does not occur in Milvus index. Index setting mismatch')
        
    def build_create_index_params(self, field_name: str) -> dict:
        params = super().build_create_index_params(field_name)
        params['params'] = {
            'n_trees': self.n_trees,
        }
        return params
    
class FloatVectorIndex_DISKANN(FloatVectorIndex, index_type='DISKANN'):...
# endregion

# region binary vector index
BinaryVectorIndexType = Literal['BIN_FLAT', 'BIN_IVF_FLAT']

class BinaryVectorIndex(MilvusIndex):
    '''Base index class for binary vector'''
    index_type: BinaryVectorIndexType
    
    def __init_subclass__(cls, index_type: BinaryVectorIndexType|None=None):
        if index_type is None:
            return
        elif not check_value_is(index_type, BinaryVectorIndexType):
            raise Exception(f'Index type of binary vector field must be one of {get_args(BinaryVectorIndexType)}')
        super().__init_subclass__(index_type)
    # TODO: implement remaining functions for binary vector index
    
# TODO: implement remaining binary vector index types
# endregion


__all__ = [
    # base class
    'MilvusIndex', 'ScalarIndex', 'VectorIndex', 
    
    # string index
    'StringIndex', 'StringIndex_MARISA_TRIE', 

    # numeric index
    'NumericIndex', 'NumericIndex_STL_SORT',
           
    # float vector index
    'FloatVectorIndex',
    'FloatVectorIndex_FLAT', 'FloatVectorIndex_IVF_FLAT', 
    'FloatVectorIndex_GPU_IVF_FLAT', 'FloatVectorIndex_IVF_SQ8', 'FloatVectorIndex_IVF_PQ', 
    'FloatVectorIndex_GPU_IVF_PQ', 'FloatVectorIndex_HNSW', 'FloatVectorIndex_SCANN', 
    'FloatVectorIndex_ANNOY', 'FloatVectorIndex_AUTOINDEX', 'FloatVectorIndex_DISKANN', 
           
    # TODO: binary vector index
    'BinaryVectorIndexType'
]


if __name__ == '__main__':
    print(_parse_anno(str|None))