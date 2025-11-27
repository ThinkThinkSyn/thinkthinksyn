if __name__ == "__main__":
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'utils.common.data_struct.advance_collections'

from enum import Enum    
from typing import Any, Callable, no_type_check, ClassVar

def _fuzzy_simplify(s: str):
    return s.strip().lower().replace(' ', '').replace('_', '').replace('-', '')

class CaseIgnoreDict(dict[str, Any]):
    '''All keys are case-insensitive, i.e. keys are saved in lower case.'''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cur_keys = tuple(self.keys())
        for k in cur_keys:
            if isinstance(k, str):
                lower_k = k.lower()
                if lower_k != k:
                    self[lower_k] = dict.pop(self, k)
    
    def __getitem__(self, key):
        key = key.lower() if isinstance(key, str) else key
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        key = key.lower() if isinstance(key, str) else key
        super().__setitem__(key, value)
        
    def __delitem__(self, key):
        key = key.lower() if isinstance(key, str) else key
        super().__delitem__(key)

    def __contains__(self, key):
        key = key.lower() if isinstance(key, str) else key
        return super().__contains__(key)
    
    def get(self, key, default=None):
        key = key.lower() if isinstance(key, str) else key
        return super().get(key, default)
    
    def pop(self, key, default=None):
        key = key.lower() if isinstance(key, str) else key
        return super().pop(key, default)
    
    def setdefault(self, key, default=None):
        key = key.lower() if isinstance(key, str) else key
        return super().setdefault(key, default)
    
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
    
    def copy(self):
        return CaseIgnoreDict(self)
    
    def __repr__(self):
        return f"<{self.__class__.__qualname__} {super().__repr__()}>"
    
    __str__ = __repr__
    
class FuzzyDict[T](dict[str, T]):
    '''
    For matching fuzzy keys, e.g. `Hello World` == `hello_world` == `helloworld`.
    Note: all keys are saved in simplified form, i.e. stripped, lower case, no space, no underscore.
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cur_keys = tuple(dict.keys(self))
        for k in cur_keys:
            if isinstance(k, str):
                simplified = _fuzzy_simplify(k)
                if simplified != k:
                    dict.__setitem__(self, simplified, dict.pop(self, k))
    
    def __getitem__(self, key):
        key = _fuzzy_simplify(key) if isinstance(key, str) else key
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        key = _fuzzy_simplify(key) if isinstance(key, str) else key
        super().__setitem__(key, value)
        
    def __delitem__(self, key):
        key = _fuzzy_simplify(key) if isinstance(key, str) else key
        super().__delitem__(key)

    def __contains__(self, key):
        key = _fuzzy_simplify(key) if isinstance(key, str) else key
        return super().__contains__(key)
    
    def get(self, key, default=None):
        key = _fuzzy_simplify(key) if isinstance(key, str) else key
        return super().get(key, default)
    
    def pop(self, key, default=None):
        key = _fuzzy_simplify(key) if isinstance(key, str) else key
        return super().pop(key, default)
    
    def setdefault(self, key, default=None):
        key = _fuzzy_simplify(key) if isinstance(key, str) else key
        return super().setdefault(key, default) # type: ignore
    
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
    
    def copy(self):
        return FuzzyDict(self)
    
    def __repr__(self):
        return f"<{self.__class__.__qualname__} {super().__repr__()}>"
    
    __str__ = __repr__
    
class ContextDict(dict[str, Any]):
    '''
    Context dict is a special dict which allows
    you to set attributes by d.key = value (equal to d['key'] = value)
    
    Specially, when you defined annotations with default values under
    the class, they will be extended to the instance's dict during
    initialization, e.g.:
    
    class A(ContextDict):
        x: int = 1  # note: names like __{..}__ will be ignored
        
    a = A()
    print(a) # {'x': 1}
    '''
    
    __ReturnValWrappers__: ClassVar[dict[str, type]] = {}
    
    def __post_init__(self):
        '''This method is called after the instance is created.
        You can override this method to do some initialization.'''
        ...
    
    @no_type_check
    class _DefaultFactory:
        def __init__(self, factory):
            self.factory = factory
        def __call__(self):
            return self.factory()
    
    @staticmethod
    def _GetReturnValWrapper(data_type: type)->type:
        data_type_name = data_type.__qualname__.split('.')[-1].split('[')[0]
        if data_type_name in ContextDict.__ReturnValWrappers__:
            return ContextDict.__ReturnValWrappers__[data_type_name]
        
        def has_attr(attr: str):
            return hasattr(data_type, attr)
        
        class _ReturnValWrapper(data_type):
            def __new__(
                cls, 
                *args,
                _context_dict: 'ContextDict' = None,    # type: ignore 
                _key: str = None,   # type: ignore
                **kwargs, 
            ):
                r = super().__new__(cls, *args, **kwargs)
                r.__context_dict__ = _context_dict   # type: ignore
                r.__key__ = _key # type: ignore
                return r
            
            def __update_self__(self):
                self.__context_dict__[self.__key__] = self
            
            if has_attr('__iadd__'):
                def __iadd__(self, other):
                    super().__iadd__(other)
                    self.__update_self__()
                    return self
            
            if has_attr('__isub__'):
                def __isub__(self, other):
                    super().__isub__(other)
                    self.__update_self__()
                    return self
                
            if has_attr('__imul__'):
                def __imul__(self, other):
                    super().__imul__(other)
                    self.__update_self__()
                    return self
                
            if has_attr('__itruediv__'):
                def __itruediv__(self, other):
                    super().__itruediv__(other)
                    self.__update_self__()
                    return self
                
            if has_attr('__ifloordiv__'):
                def __ifloordiv__(self, other):
                    super().__ifloordiv__(other)
                    self.__update_self__()
                    return self
                
            if has_attr('__imod__'):
                def __imod__(self, other):
                    super().__imod__(other)
                    self.__update_self__()
                    return self
            
            if has_attr('__ipow__'):
                def __ipow__(self, other):
                    super().__ipow__(other)
                    self.__update_self__()
                    return self
            
            if has_attr('__iand__'):
                def __iand__(self, other):
                    super().__iand__(other)
                    self.__update_self__()
                    return self
                
            if has_attr('__ior__'):
                def __ior__(self, other):
                    super().__ior__(other)
                    self.__update_self__()
                    return self
                
            if has_attr('__ixor__'):
                def __ixor__(self, other):
                    super().__ixor__(other)
                    self.__update_self__()
                    return self    
                
        ContextDict.__ReturnValWrappers__[data_type_name] = _ReturnValWrapper
        return _ReturnValWrapper
        
    @staticmethod
    @no_type_check
    def DefaultFactory(factory: Callable[[], Any])->Any:
        '''
        Define a default factory for the attribute,
        e.g.:
        ```
        class A(ContextDict):
            x: int = ContextDict.DefaultFactory(lambda: 1)
        ```
        '''
        return ContextDict._DefaultFactory(factory) # type: ignore
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls_annos = get_cls_annotations(self.__class__)
        for k in cls_annos:
            if k.startswith('__') and k.endswith('__'):
                continue
            if k in self.__class__.__dict__:  # has default value
                if not dict.__contains__(self, k):
                    data = self.__class__.__dict__[k]
                    if isinstance(data, ContextDict._DefaultFactory):
                        data = data()
                    self[k] = data
        self.__post_init__()
        
    def __getattribute__(self, name: str) -> Any:
        if not name.startswith('__'):
            if name in self:
                val = self[name]
                if isinstance(val, (int, float, str, bytes, bool)) and not isinstance(val, Enum):
                    wrapper = ContextDict._GetReturnValWrapper(type(val))
                    return wrapper(val, _context_dict=self, _key=name)  # type: ignore
                return val
        return super().__getattribute__(name)
    
    def __setattr__(self, key, value):
        if not key.startswith('__'):
            if (key not in dict.__dict__) and (key not in ContextDict.__dict__) and (key not in self.__dict__):
                self[key] = value
                return
        super().__setattr__(key, value)
    
    def __repr__(self):
        return f"<{self.__class__.__qualname__} {' '.join(f'{k}={v!r}' for k, v in self.items())}>"
    
    __str__ = __repr__
    

__all__ = ['CaseIgnoreDict', 'FuzzyDict', 'ContextDict']


if __name__ == '__main__':
    def test_context_dict():
        class A(ContextDict):
            x: int = 1
            z: int = ContextDict.DefaultFactory(lambda: 1)
        
        a = A(x=2, y=2)
        print(a, dict(a), a.x)
        a.x += 10
        print(a, a.x)
        a.w = 1
        print(a, a.w)
        a.w += 1
        print(a, a.w)
        
    def test_case_ignore_dict():
        d = CaseIgnoreDict()
        d['Hello'] = 'World'
        print(d['hello'])
        print(d.get('Hello'))
        
    # test_context_dict()
    test_case_ignore_dict()