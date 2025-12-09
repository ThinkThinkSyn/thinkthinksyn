'''
Class property makes the property workable for the class instead of the instance.
Note: `setter` is not supported for class property, due to the limitation of python.
'''

from collections.abc import Callable
from functools import update_wrapper
from typing import (Type, override, Generic, TypeVar, overload, Union, Self, Concatenate, ParamSpec)


class _class_property(property): # still inherit from property(but it works nothing for this class), cuz it makes `isinstance(..., property)` True
    
    def __init__(self, fget):
        self.getter = fget
        update_wrapper(self, fget)  # type: ignore
    
    def __get__(self, _, owner):
        return self.getter(owner)

_ClsT = TypeVar('_ClsT')
_RetT = TypeVar('_RetT')

class class_property(Generic[_ClsT, _RetT]): # this just for type hint
    '''
    Class property decorator. Acts like @property but for the class instead.
    Due to the limitation of python, @setter is not supported for class property.
    
    Example:
    ```python
    from typing import Type, Self
    class A:
        def __init__(self, k):
            self.k = k
        @class_property
        def a(cls:Type[Self]):
            return cls(k=1)
    
    print(A.a.k)  # 1
    ''' 
    @overload
    def __new__(self, fget:Callable[[Type[_ClsT]], _RetT])->Self: ... # type: ignore
    @overload
    def __new__(self, fget:Callable[..., _RetT])->Self: ...
    
    def __get__(self, instance, owner: Type[_ClsT])->_RetT: ... # type: ignore
    
globals()['class_property'] = _class_property

class _abstract_class_property(_class_property):
    __isabstractmethod__ = True
class abstract_class_property(class_property): # this just for type hint
    '''
    Abstract class property decorator. Acts like @property but for the class instead.
    
    Example:
    ```python
    class A(ABC):
        @abstract_class_property
        def prop(cls):
            raise NotImplementedError()
    '''

globals()['abstract_class_property'] = _abstract_class_property

class _constant_cls_property(_class_property):
    __property_name__: str|None = None
    
    def __set_name__(self, owner, name):
        self.__property_name__ = name
        
    def __get__(self, _, owner):
        if not self.__property_name__:
            return self.getter(owner)
        else:
            cache_name = f'__{self.__property_name__}__'
            if cache_name in owner.__dict__:
                return owner.__dict__[cache_name]
            else:
                r = self.getter(owner)
                setattr(owner, cache_name, r)
                return r

class constant_cls_property(class_property): # this just for type hint
    '''
    Constant class property decorator. Acts like @property but for the class instead.
    The property value will be cached after first access.
    
    Example:
    ```python
    class A:
        @constant_cls_property
        def a(cls):
            return 1
    '''

globals()['constant_cls_property'] = _constant_cls_property

class _class_or_ins_property(property): # still inherit from property(but it works nothing for this class), cuz it makes `isinstance(..., property)` True
    def __init__(self, fget):
        self.getter = fget  # type: ignore
        update_wrapper(self, fget)  # type: ignore
        
    def __get__(self, instance, owner):
        if instance is None:
            return self.getter(owner)   # type: ignore
        else:
            return self.getter(instance)    # type: ignore

class class_or_ins_property(Generic[_ClsT, _RetT]):   # this just for type hint
    '''
    Decorator for class or instance property. Acts like @property but also workable for the class instead.
    
    Example:
    ```python
    class A:
        @class_or_ins_method
        def func(cls_or_self):
            print(cls_or_self)
    
    A.func() # <class '__main__.A'>
    A().func() # <__main__.A object at ....>
    ```
    '''
    @overload
    def __new__(self, fget:Callable[[Type[_ClsT]], _RetT])->Self: ... # type: ignore
    @overload
    def __new__(self, fget:Callable[..., _RetT])->Self: ...
    @override
    def __get__(self, instance, owner:Type[_ClsT])->_RetT: ...   # type: ignore
    
globals()['class_or_ins_property'] = _class_or_ins_property

class _class_or_ins_method(classmethod):
    def __get__(self, instance, owner):
        if not instance:
            return super().__get__(owner, owner)    # type: ignore
        else:
            return super().__get__(owner, instance)   # type: ignore

_T = TypeVar('_T')
_R = TypeVar('_R')
_P = ParamSpec('_P')

class class_or_ins_method(Generic[_T,_P,_R]):   # this just for type hint
    '''
    When calling without instance, the first param will be the class itself.
    If called with instance, the first param will be the instance itself.
    '''
    
    @overload
    def __new__(self, f: Callable[Concatenate[_T, _P], _R])->Self:...  # type: ignore
    def __get__(self, ins, owner)->Self:...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs)->_R:...

globals()['class_or_ins_method'] = _class_or_ins_method


__all__ = ['class_property', 'abstract_class_property', 'constant_cls_property', 
           'class_or_ins_method', 'class_or_ins_property']


def prevent_re_init(cls: _ClsT)->_ClsT:
    '''
    Prevent re-init of a created instance.
    This decorator is useful when you want to make sure that the instance is only created once, e.g. singleton.
    '''
    
    origin_init = cls.__init__
    def new_init(self, *args, **kwargs):
        if hasattr(self, '__prevent_re_inited__') and getattr(self, '__prevent_re_inited__'):
            return  # do nothing
        setattr(self, '__prevent_re_inited__', True)
        if origin_init == object.__init__:
            origin_init(self)   # type: ignore
        else:
            origin_init(self, *args, **kwargs) # type: ignore
    cls.__init__ = new_init # type: ignore
    return cls

@overload
def singleton(cls: _ClsT)->_ClsT:   # type: ignore
    '''
    The singleton class decorator.
    Singleton class is a class that only has one instance. 
    You will still got the same instance even if you create the instance multiple times.
    
    Note: `singleton` is available for `attrs` class, but not available for pydantic BaseModel.
    '''
@overload
def singleton(cross_module_singleton: bool=True)->Callable[[ClsT], ClsT]:   # type: ignore
    '''
    The singleton class decorator.
    Singleton class is a class that only has one instance. 
    You will still got the same instance even if you create the instance multiple times.
    
    if `cross_module_singleton` is True, the class will be unique even if you import the module in different ways.
    This is helpful when the import relationship is complex.
    
    Note: `singleton` is available for `attrs` class, but not available for pydantic BaseModel.
    '''

_cross_module_cls_dict = dict()

def _singleton(cls: _ClsT, cross_module_singleton:bool=False)->_ClsT:
    '''
    Make the class as singleton class.
    Singleton class is a class that only has one instance. 
    You will still got the same instance even if you create the instance multiple times.
    '''
    if hasattr(cls, '__singleton_decorated__'):
        return cls
    if cross_module_singleton:
        cls_name = cls.__qualname__
        if cls_name in _cross_module_cls_dict:
            return _cross_module_cls_dict[cls_name].__singleton_instance__
    
    origin_new = cls.__new__
    def new_new(this_cls, *args, **kwargs):
        if hasattr(this_cls, '__singleton_instance__') and getattr(this_cls, '__singleton_instance__') is not None:
            return getattr(this_cls, '__singleton_instance__')
        if origin_new is object.__new__:
            return origin_new(this_cls) # type: ignore
        return origin_new(this_cls, *args, **kwargs)
    cls.__new__ = new_new   # type: ignore
    
    origin_init = cls.__init__
    def new_init(self, *args, **kwargs):
        if hasattr(self.__class__, '__singleton_instance__') and getattr(self.__class__, '__singleton_instance__') is not None:
            return  # do nothing
        setattr(self.__class__, '__singleton_instance__', self)
        origin_init(self, *args, **kwargs)  # type: ignore
    cls.__init__ = new_init  # type: ignore
    
    if cross_module_singleton:
        _cross_module_cls_dict[cls_name] = cls
    setattr(cls, '__singleton_decorated__', True)
    return cls

def singleton(*args, **kwargs):  # type: ignore
    if len(args) + len(kwargs) != 1:
        raise ValueError('singleton only accept one argument.')
    
    arg = args[0] if args else None
    if arg is None:
        for _, v in kwargs.items():
            arg = v
            break
    
    if isinstance(arg, bool):
        return lambda cls: _singleton(cls, arg)
    else:
        return _singleton(arg)  # type: ignore


__all__.extend(['prevent_re_init', 'singleton'])


if __name__ == '__main__':  # for debugging
    from attr import attrs
    @singleton
    @attrs(auto_attribs=True)
    class A:
        x: int
        
    a1 = A(1)
    a2 = A(2)
    print(a1 is a2)  # True
    print(type(a1), type(a2))  # <class '__main__.A'> <class '__main__.A'>
    print(a1.x, a2.x)  # 1 1