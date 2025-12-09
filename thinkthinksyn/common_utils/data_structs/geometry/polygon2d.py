import numpy as np

from typing import overload, Sequence
from pydantic_core import core_schema

from .point2d import Point2D


class Polygon2D(tuple[Point2D, ...]):
    '''dots to bound a region. Dots should be in clockwise.'''
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validator(data):
            if isinstance(data, cls):
                return data
            if isinstance(data, (list, tuple)):
                return cls(data)
            if isinstance(data, np.ndarray):
                assert data.ndim==2, f'Expect 2 dimensions for numpy array, but got {data.ndim}.'
                assert data.shape[1]==2, f'Expect 2 elements for the second dimension, but got {data.shape[1]} elements.'
                return cls([Point2D(i) for i in data])
            raise ValueError(f'Unexpected data type: {type(data)}')
        
        schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=schema,
        )
        
    @overload
    def __new__(cls, *points: Point2D):...
    @overload
    def __new__(cls, points: Sequence[Point2D|list[float]|tuple[float, float]]):...
    
    def __new__(cls, *args):  # type: ignore
        if len(args)==1:
            args = args[0]
        proper_args = list(args)
        for i in range(len(proper_args)):
            if not isinstance(proper_args[i], Point2D):
                assert len(proper_args[i])==2, f'Expect 2 elements for Point2D, but got {len(proper_args[i])} elements.'
                proper_args[i] = Point2D(proper_args[i])
        assert len(proper_args)>=3, f'Expect at least 3 points to bound a region, but got {len(proper_args)} points.'
        return super().__new__(cls, proper_args)
    
    def __repr__(self):
        return f'Polygon2D({", ".join(map(str, self))})'
    
    def __eq__(self, a):
        if isinstance(a, (list, tuple)):
            return all([i==j for i, j in zip(self, a)])
        return super().__eq__(a)
    
    

__all__ = ['Polygon2D']
