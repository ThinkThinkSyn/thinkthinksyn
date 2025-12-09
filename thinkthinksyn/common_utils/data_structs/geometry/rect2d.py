import numpy as np

from typing import overload
from pydantic_core import core_schema

from .point2d import Point2D
from .line2d import Line2D


class Rect2D(tuple[Point2D, Point2D, Point2D, Point2D]):
    '''top left, top right, bottom right, bottom left (clockwise)'''
    
    @overload
    def __new__(cls, top_left: Point2D, top_right: Point2D, bottom_right: Point2D, bottom_left: Point2D):...
    @overload
    def __new__(cls, rect: list[Point2D]|tuple[Point2D, Point2D, Point2D, Point2D]):...
    def __new__(cls, *args):  # type: ignore
        if len(args)==1:
            args = args[0]
        proper_args = list(args)
        for i in range(4):
            if not isinstance(proper_args[i], Point2D):
                assert len(proper_args[i])==2, f'Expect 2 elements for Point2D, but got {len(proper_args[i])} elements.'
                proper_args[i] = Point2D(proper_args[i])
        return super().__new__(cls, proper_args)  # type: ignore
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validator(data):
            if isinstance(data, cls):
                return data
            if isinstance(data, (list, tuple)):
                if len(data)==4:
                    return cls(data)    # type: ignore
                raise ValueError(f'Expect 4 elements in the list or tuple, but got {len(data)} elements.')
            if isinstance(data, dict):
                if 'top_left' in data and 'top_right' in data and 'bottom_right' in data and 'bottom_left' in data:
                    return cls(data['top_left'], data['top_right'], data['bottom_right'], data['bottom_left'])
                raise ValueError(f'Expect keys "top_left", "top_right", "bottom_right", "bottom_left" in the dict, but got {list(data.keys())}.')
            if isinstance(data, np.ndarray):
                if data.shape==(4, 2):
                    return cls(data[0], data[1], data[2], data[3])
                raise ValueError(f'Expect shape (4, 2) for numpy array, but got {data.shape}.')
            raise ValueError(f'Unexpected data type: {type(data)}')
        
        schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=schema,
        )
    
    @property
    def top_left(self)->Point2D:
        return self[0]
    @property
    def top_right(self)->Point2D:
        return self[1]
    @property
    def bottom_right(self)->Point2D:
        return self[2]
    @property
    def bottom_left(self)->Point2D:
        return self[3]
    
    @property
    def horizon_middle_line(self)->Line2D:
        return Line2D((self.top_left + self.bottom_left)/2, 
                      (self.top_right + self.bottom_right)/2)
    @property
    def vertical_middle_line(self)->Line2D:
        return Line2D((self.top_left + self.top_right)/2, 
                      (self.bottom_left + self.bottom_right)/2)
    
    @property
    def width(self)->float:
        return ((self.top_right.x-self.top_left.x) + (self.bottom_right.x-self.bottom_left.x))/2
    @property
    def height(self)->float:
        return ((self.bottom_left.y-self.top_left.y) + (self.bottom_right.y-self.top_right.y))/2
    @property
    def center(self)->Point2D:
        return Point2D((self.top_left.x+self.width/2, self.top_left.y+self.height/2))
    
    @property
    def left_line(self)->Line2D:
        return Line2D((self.top_left, self.bottom_left))
    @property
    def right_line(self)->Line2D:
        return Line2D((self.top_right, self.bottom_right))
    @property
    def top_line(self)->Line2D:
        return Line2D((self.top_left, self.top_right))
    @property
    def bottom_line(self)->Line2D:
        return Line2D((self.bottom_left, self.bottom_right))
    
    def __repr__(self):
        return f'Rect2D({self.top_left}, {self.top_right}, {self.bottom_right}, {self.bottom_left})'
    def __eq__(self, a):
        if isinstance(a, Rect2D):
            return self.top_left==a.top_left and self.top_right==a.top_right and self.bottom_right==a.bottom_right and self.bottom_left==a.bottom_left
        return False
    def __add__(self, a):
        if isinstance(a, Point2D):
            return Rect2D((self.top_left+a, self.top_right+a, self.bottom_right+a, self.bottom_left+a))
        return Rect2D((self.top_left+a, self.top_right+a, self.bottom_right+a, self.bottom_left+a))
    def __sub__(self, a):
        if isinstance(a, Point2D):
            return Rect2D((self.top_left-a, self.top_right-a, self.bottom_right-a, self.bottom_left-a))
        return Rect2D((self.top_left-a, self.top_right-a, self.bottom_right-a, self.bottom_left-a))
    def __mul__(self, a):
        if isinstance(a, Point2D):
            return Rect2D((self.top_left*a, self.top_right*a, self.bottom_right*a, self.bottom_left*a))
        return Rect2D((self.top_left*a, self.top_right*a, self.bottom_right*a, self.bottom_left*a))

    
    

__all__ = ['Rect2D']

