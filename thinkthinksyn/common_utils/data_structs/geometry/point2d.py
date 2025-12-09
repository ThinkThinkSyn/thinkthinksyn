import numpy as np

from typing import overload
from pydantic_core import core_schema

class Point2D(tuple[float, float]):

    @overload
    def __new__(cls, x: float, y: float):...
    @overload
    def __new__(cls, xy: tuple[float|int, float|int]):...
    def __new__(cls, *args):  # type: ignore
        if len(args)==1:
            if isinstance(args[0], (list, tuple, np.ndarray)):
                args = args[0]
            elif isinstance(args[0], dict):
                args = args[0].values()
            else:
                raise ValueError(f'Unexpected data type: {type(args[0])}')
        assert len(args)==2, f'Expect 2 elements for Line2D, but got {len(args)} elements.'
        args = [float(i) for i in args]
        return super().__new__(cls, args)  # type: ignore
    
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
                if 'x' in data and 'y' in data:
                    return cls(data['x'], data['y'])
                raise ValueError(f'Expect keys "x" and "y" in the dict, but got {list(data.keys())}.')
            if isinstance(data, np.ndarray):
                if data.shape==(2,):
                    return cls(data[0], data[1])
                raise ValueError(f'Expect shape (2,) for numpy array, but got {data.shape}.')
            raise ValueError(f'Unexpected data type: {type(data)}')
        
        schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=schema,
        )
            
    @property
    def x(self)->float:
        return self[0]
    @property
    def y(self)->float:
        return self[1]
    
    def distance(self, other: "Point2D")->float:
        return ((self.x-other.x)**2+(self.y-other.y)**2)**0.5
    
    def __add__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return cls((self.x+a.x, self.y+a.y))
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return cls((self.x+a[0], self.y+a[1]))
        return cls((self.x+a, self.y+a))
    
    def __sub__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return cls((self.x-a.x, self.y-a.y))
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return cls((self.x-a[0], self.y-a[1]))
        return cls((self.x-a, self.y-a))
    
    def __mul__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return cls((self.x*a.x, self.y*a.y))
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return cls((self.x*a[0], self.y*a[1]))
        return cls((self.x*a, self.y*a))
    
    def __truediv__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return cls((self.x/a.x, self.y/a.y))
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return cls((self.x/a[0], self.y/a[1]))
        return cls((self.x/a, self.y/a))
    
    def __floordiv__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return cls((self.x//a.x, self.y//a.y))
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return cls((self.x//a[0], self.y//a[1]))
        return cls((self.x//a, self.y//a))
    
    def __mod__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return cls((self.x%a.x, self.y%a.y))
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return cls((self.x%a[0], self.y%a[1]))
        return cls((self.x%a, self.y%a))
    
    def __pow__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return cls((self.x**a.x, self.y**a.y))
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return cls((self.x**a[0], self.y**a[1]))
        return cls((self.x**a, self.y**a))
    
    def __abs__(self):
        cls = self.__class__
        return cls((abs(self.x), abs(self.y)))
    
    def __neg__(self):
        return Point2D((-self.x, -self.y))
    
    def __eq__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return self.x==a.x and self.y==a.y
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return self.x==a[0] and self.y==a[1]
        return False
    
    def __lt__(self, a):
        cls = self.__class__
        if isinstance(a, cls):
            return self.x<a.x and self.y<a.y
        elif isinstance(a, (tuple, list)):
            assert len(a)==2, f'Expect 2 elements for {cls.__qualname__.split('.')[-1]}, but got {len(a)} elements.'
            return self.x<a[0] and self.y<a[1]
        return False
    
    def __repr__(self):
        cls = self.__class__
        return f'{cls.__qualname__.split('.')[-1]}({self.x}, {self.y})'


__all__ = ['Point2D']


if __name__ == '__main__':
    print(max(Point2D(0.2,0.2), Point2D(0.3,0.3)))