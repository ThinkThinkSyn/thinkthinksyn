import numpy as np

from typing import overload, Literal, Callable
from pydantic_core import core_schema

from .point2d import Point2D


class Line2D(tuple[Point2D, Point2D]):
    
    @overload
    def __new__(cls, start: Point2D, end: Point2D):...
    @overload
    def __new__(cls, line: tuple[Point2D, Point2D]):...
    def __new__(cls, *args):  # type: ignore
        if len(args)==1:
            args = args[0]
        assert len(args)==2, f'Expect 2 elements for Line2D, but got {len(args)} elements.'
        proper_args = list(args)
        for i in range(2):
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
                if len(data)==2:
                    return cls(data)    # type: ignore
                raise ValueError(f'Expect 2 elements in the list or tuple, but got {len(data)} elements.')
            if isinstance(data, dict):
                if 'start' in data and 'end' in data:
                    return cls(data['start'], data['end'])
                elif 'from' in data and 'to' in data:
                    return cls(data['from'], data['to'])
                raise ValueError(f'Expect keys "start" and "end" in the dict, but got {list(data.keys())}.')
            if isinstance(data, np.ndarray):
                if data.shape==(2, 2):
                    return cls(data[0], data[1])
                raise ValueError(f'Expect shape (2, 2) for numpy array, but got {data.shape}.')
            raise ValueError(f'Unexpected data type: {type(data)}')
        
        schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=schema,
        )
        
    @property
    def start(self)->Point2D:
        return self[0]
    @property
    def end(self)->Point2D:
        return self[1]
    
    @property
    def slope(self)->float|None:
        '''return None if the line is vertical.'''
        start = self.start
        end = self.end
        if end.x==start.x:
            return None
        return (end.y-start.y)/(end.x-start.x)
    
    @property
    def length(self)->float:
        start = self.start
        end = self.end
        return ((end.x-start.x)**2+(end.y-start.y)**2)**0.5
    
    @property
    def middle(self)->Point2D:
        start = self.start
        end = self.end
        return Point2D(((start.x+end.x)/2, (start.y+end.y)/2))
    
    def intersect(self, 
                  other: "Line2D", 
                  mode: Literal['within', 'extend_self', 'extend_other', 'extend_both']='extend_self')->Point2D|None:
        '''
        Get the intersection point of two lines. 
        Return None if they are parallel or no intersection point.
        
        Args:
            other: the other line to intersect with.
            mode:
                'within': only return the intersection point within the range of both lines.
                'extend_self': extend the self line to intersect with the other line.
                'extend_other': extend the other line to intersect with the self line.
                'extend_both': extend both lines to intersect.
        '''
        self_slope = self.slope
        other_slope = other.slope
        
        if self_slope==other_slope:
            return None
        self_start = self.start
        other_start = other.start
        
        if self_slope is None:
            x = self_start.x
            y = other_slope*(x-other_start.x)+other_start.y    # type: ignore
        elif other_slope is None:
            x = other_start.x
            y = self_slope*(x-self_start.x)+self_start.y
        else: 
            if self_slope==0:
                y = self_start.y
                x = (y-other_start.y)/other_slope+other_start.x
            elif other_slope==0:
                y = other_start.y
                x = (y-self_start.y)/self_slope+self_start.x
            else:
                x = (self_slope*self_start.x-other_slope*other_start.x+other_start.y-self_start.y)/(self_slope-other_slope)
                y = self_slope*(x-self_start.x)+self_start.y
                
        if mode=='within':
            if x<min(self_start.x, self.end.x) or x>max(self_start.x, self.end.x) or \
                x<min(other_start.x, other.end.x) or x>max(other_start.x, other.end.x):
                return None # out of range
            if y<min(self_start.y, self.end.y) or y>max(self_start.y, self.end.y) or \
                y<min(other_start.y, other.end.y) or y>max(other_start.y, other.end.y):
                return None
        elif mode=='extend_self':
            if x<min(other_start.x, other.end.x) or x>max(other_start.x, other.end.x):
                return None
            if y<min(other_start.y, other.end.y) or y>max(other_start.y, other.end.y):
                return None
        elif mode=='extend_other':
            if x<min(self_start.x, self.end.x) or x>max(self_start.x, self.end.x):
                return None
            if y<min(self_start.y, self.end.y) or y>max(self_start.y, self.end.y):
                return None
        return Point2D((x, y))
    
    @property
    def vector(self):
        from .vec2d import Vec2D
        return Vec2D(self.end-self.start)
    
    @property
    def formula_x(self)->Callable[[float|int], float]:
        '''return f(x)->y'''
        start = self.start
        slope = self.slope
        if slope is None:
            raise ValueError('Vertical line has no f(x)->y formula.')
        return lambda x: slope*(x-start.x)+start.y
    
    @property
    def formula_y(self)->Callable[[float|int], float]:
        '''return f(y)->x'''
        start = self.start
        slope = self.slope
        if slope == 0:
            raise ValueError('Horizontal line has no f(y)->x formula.')
        if slope is None:
            return lambda y: start.x
        return lambda y: (y-start.y)/slope+start.x
    
    def __repr__(self):
        cls = self.__class__
        return f'{cls.__qualname__.split('.')[-1]}({self.start}, {self.end})'
    
    def __eq__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return self.start==a.start and self.end==a.end
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert isinstance(a[0], (tuple, list)) and len(a[0])==2, f"Expect 2 elements for comparing with the line's first point, but got {len(a[0])} elements."
            assert isinstance(a[1], (tuple, list)) and len(a[1])==2, f"Expect 2 elements for comparing with the line's first point, but got {len(a[1])} elements."
            return self.start==a[0] and self.end==a[1]
        return False
    
    def __add__(self, a):
        cls = self.__class__
        if isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert len(a[0]) == len(a[1]), f'Expect 2 same elements for each point, but got {len(a[0])} elements for the first point and {len(a[1])} elements for the second point.'
            assert len(a[0]) <=2, f'Expect 1 or 2 elements for each point(can be a point or line), but got {len(a[0])} elements for point.'
            return cls((self.start+a[0], self.end+a[1]))
        return cls((self.start+a, self.end+a))
    
    def __sub__(self, a):
        cls = self.__class__
        if isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert len(a[0]) == len(a[1]), f'Expect 2 same elements for each point, but got {len(a[0])} elements for the first point and {len(a[1])} elements for the second point.'
            assert len(a[0]) <=2, f'Expect 1 or 2 elements for each point(can be a point or line), but got {len(a[0])} elements for point.'
            return cls((self.start-a[0], self.end-a[1]))
        return cls((self.start-a, self.end-a))
    
    def __mul__(self, a):
        cls = self.__class__
        if isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert len(a[0]) == len(a[1]), f'Expect 2 same elements for each point, but got {len(a[0])} elements for the first point and {len(a[1])} elements for the second point.'
            assert len(a[0]) <=2, f'Expect 1 or 2 elements for each point(can be a point or line), but got {len(a[0])} elements for point.'
            return cls((self.start*a[0], self.end*a[1]))
        return cls((self.start*a, self.end*a))
    
    def __truediv__(self, a):
        cls = self.__class__
        if isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert len(a[0]) == len(a[1]), f'Expect 2 same elements for each point, but got {len(a[0])} elements for the first point and {len(a[1])} elements for the second point.'
            assert len(a[0]) <=2, f'Expect 1 or 2 elements for each point(can be a point or line), but got {len(a[0])} elements for point.'
            return cls((self.start/a[0], self.end/a[1]))
        return cls((self.start/a, self.end/a))
    
    def __floordiv__(self, a):
        cls = self.__class__
        if isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert len(a[0]) == len(a[1]), f'Expect 2 same elements for each point, but got {len(a[0])} elements for the first point and {len(a[1])} elements for the second point.'
            assert len(a[0]) <=2, f'Expect 1 or 2 elements for each point(can be a point or line), but got {len(a[0])} elements for point.'
            return cls((self.start//a[0], self.end//a[1]))
        return cls((self.start//a, self.end//a))
    
    def __mod__(self, a):
        cls = self.__class__
        if isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert len(a[0]) == len(a[1]), f'Expect 2 same elements for each point, but got {len(a[0])} elements for the first point and {len(a[1])} elements for the second point.'
            assert len(a[0]) <=2, f'Expect 1 or 2 elements for each point(can be a point or line), but got {len(a[0])} elements for point.'
            return cls((self.start%a[0], self.end%a[1]))
        return cls((self.start%a, self.end%a))
        
    def __pow__(self, a):
        cls = self.__class__
        if isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            assert len(a[0]) == len(a[1]), f'Expect 2 same elements for each point, but got {len(a[0])} elements for the first point and {len(a[1])} elements for the second point.'
            assert len(a[0]) <=2, f'Expect 1 or 2 elements for each point(can be a point or line), but got {len(a[0])} elements for point.'
            return cls((self.start**a[0], self.end**a[1]))
        return cls((self.start**a, self.end**a))
    
    def __abs__(self):
        cls = self.__class__
        return cls((abs(self.start), abs(self.end)))


__all__ = ['Line2D']