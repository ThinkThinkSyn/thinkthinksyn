import re
import inspect 

from functools import partial
from collections import OrderedDict
from dataclasses import dataclass
from inspect import Parameter, signature, _empty
from typing import Any, Callable, overload, Sequence, Literal
from types import MethodType, FunctionType, BuiltinFunctionType

@dataclass
class PackFuncParamOutput:
    '''
    The output of function `pack_param`.
    The return value includes some useful information, for further usage.
    '''
    packed_params: OrderedDict[str, Any]
    '''The packed params, e.g. {'a': 1, 'b': (2, 3), 'c': {'c1': 1, 'c2': 2}}'''
    func_params: OrderedDict[str, Parameter]
    '''The parameters of the function, e.g. {'a': <Parameter "a:int">, ...}, see inspect.Parameter for details'''
    var_args_field_name: str|None
    '''the `*args` field name. e.g. def f(*a) -> var_args_field_name="a"'''
    var_kwargs_field_name: str|None
    '''the `**kwargs` field name. e.g. def f(**a) -> var_kwargs_field_name="a"'''
    
    def to_func_params(self)->tuple[tuple, dict[str, Any]]:
        '''
        Convert the packed params to a tuple and a dict, for further usage.
        e.g.:
        ```python
        packed_params = PackFuncParamOutput(...)
        args, kwargs = packed_params.to_func_params()
        ```
        '''
        args = []
        kwargs = {}
        next_kw_only = False
        
        def add_val(pn, p, val):
            nonlocal next_kw_only
            if p.kind == Parameter.VAR_POSITIONAL:
                args.extend(val)
                next_kw_only = True
            elif p.kind == Parameter.VAR_KEYWORD:
                kwargs.update(val)
                next_kw_only = True
            elif p.kind == Parameter.POSITIONAL_ONLY:
                args.append(val)
            elif p.kind == Parameter.KEYWORD_ONLY:
                kwargs[pn] = val
                next_kw_only = True
            else:
                if next_kw_only:
                    kwargs[pn] = val
                else:
                    args.append(val)
        
        for pn, p in self.func_params.items():
            if pn in self.packed_params:
                val = self.packed_params[pn]
                add_val(pn, p, val)
            elif p.default != _empty:
                val = p.default
                add_val(pn, p, val)
            else:
                raise TypeError(f'Missing required parameter: {pn}')
        
        return tuple(args), kwargs

@overload
def pack_function_params(origin_func: Callable, args: Sequence[Any], kwargs: dict[str, Any], return_detail: Literal[True] = True)-> PackFuncParamOutput:...
@overload
def pack_function_params(origin_func: Callable, args: Sequence[Any], kwargs: dict[str, Any], return_detail: Literal[False]=False)-> OrderedDict[str, Any]:...

def pack_function_params(origin_func: Callable, args: Sequence[Any], kwargs: dict[str, Any], return_detail: bool=True):
    '''
    Pack the args & kwargs to the format of the origin function.
    The return value includes some useful information. See `PackParamOutput` for details.
    
    e.g.
    ```python
    def f(a, *b, **c):
        pass
    print(pack_param(f, ('a', 'b1', 'b2'), {c1=1, c2=2}).packed_params) # {'a': 'a', 'b': ('b1', 'b2'), 'c': {'c1': 1, 'c2': 2}}
    ```
    '''
    sig = signature(origin_func)
    bound = OrderedDict(sig.bind(*args, **kwargs).arguments)
    if not return_detail:
        return bound
    
    params = OrderedDict(sig.parameters)
    var_args_field_name = None
    var_kwargs_field_name = None

    for param_name, param in params.items():
        if param.kind == Parameter.VAR_POSITIONAL:
            var_args_field_name = param_name
        elif param.kind == Parameter.VAR_KEYWORD:
            var_kwargs_field_name = param_name
        if var_args_field_name and var_kwargs_field_name:
            break
    
    return PackFuncParamOutput(
        packed_params=bound, 
        func_params=OrderedDict(sig.parameters), 
        var_args_field_name=var_args_field_name,
        var_kwargs_field_name=var_kwargs_field_name
    )


def func_param_type_check(func:Callable, *args, **kwargs)->bool:
    '''
    Check if the args and kwargs of func are valid.
    E.g.:
    ```
    def func(a:int, b:str, c):
        ...

    func_param_type_check(func, 1, 'abc', c=1.0) 
    # True, for "c", since no type is specified, it will be Any
    ```
    '''
    # pack all args and kwargs into a dict
    try:
        pack_function_params(func, args, kwargs)
    except TypeError:
        return False
    return True

_empty_method_cache: dict[str, bool] = {}

def is_empty_method(method: Callable):
    '''
    Check if a method is totally empty, i.e. no logic code in the method, except for:
        * docstring/comment
        * `pass`
        * `raise NotImplementedError`
    '''
    if isinstance(method, BuiltinFunctionType):
        return False
    if not (callable(method) or isinstance(method, (classmethod, staticmethod, MethodType, FunctionType, partial))):
        raise TypeError(f'Expected a callable, but got `{type(method)}`')
    
    method_name = get_func_name(method, no_module=False)
    if method_name in _empty_method_cache:
        return _empty_method_cache[method_name]
    
    is_empty = False
    if hasattr(method, '__doc__'):
        doc_str = method.__doc__
    else:
        doc_str = None
    try:
        source = inspect.getsource(method)
    except Exception as e:
        # fail to get source, e.g. built-in functions
        source = None
    
    if source is not None:
        if doc_str:
            source = source.replace(doc_str, '')
        
        func_def_pattern = re.compile(r'(async)?\s*def\s+\w+\s*\(.*\).*?:', re.MULTILINE|re.DOTALL)
        source = re.sub(func_def_pattern, '', source, count=1)
        lines = source.split('\n')
        for l in lines:
            l = l.strip()
            # remove empty lines
            if not l:
                continue
            # remove comment/pass lines
            if not (not l.startswith(('#', '"""',"'''")) and not l in ('pass', '...')):
                continue
            # remove `raise NotImplementedError` lines 
            no_implement_error_pattern = r'raise\s+NotImplementedError(?:\(\s*\))?\s*'
            if re.match(no_implement_error_pattern, l):
                continue
            # remove `return` lines
            return_pattern = r'\s*return\s*|\s*return\s+None\s*'
            if re.match(return_pattern, l):
                continue
            break   # any valid line
        else:
            is_empty = True
    
    _empty_method_cache[method_name] = is_empty
    return is_empty

def func_arg_count(
    func: Callable|classmethod|staticmethod,
    ignore_cls: bool=False,
):
    '''
    Get the count of arguments of a function.
    Args:
        - func: the function to check
        - ignore_cls: if True, ignore first param when `func` is a classmethod
    '''
    origin_func = func
    if isinstance(func, (classmethod, staticmethod)):
        func = func.__func__
    c = len(signature(func).parameters)
    if ignore_cls and isinstance(origin_func, classmethod):
        c -= 1
    return c

def get_func_name(
    func: Callable|classmethod|staticmethod, 
    no_module: bool=False,
)->str:
    '''
    Get the name of a function.
    
    If this given function doesn't have both `__qualname__` and `__name__` attribute,
    return a string representation of the function(special chars are removed and spaces are
    turned into underscores).
    
    if `no_module` is True, the module name(s) will be removed,
    i.e. name.split('.')[-1] will be returned.
    '''
    module = None
    if not no_module and hasattr(func, '__module__'):
        from .type_helpers import get_module_name
        module = get_module_name(func)
    
    while isinstance(func, (classmethod, staticmethod, MethodType)):
        func = func.__func__
    if isinstance(func, partial):
        func = func.func
    
    if hasattr(func, '__qualname__'):
        name = func.__qualname__
    elif hasattr(func, '__name__'):
        name = func.__name__
    else:
        pattern = r'(?:function|method)\s(.*)\sat'
        if m:=re.search(pattern, str(func)):
            name = m.group(1)
        else:
            name =str(func).replace('<', '').replace('>', '').replace(' ', '_')
    
    if module:
        return f'{module}.{name}'
    return name
    
def unpack_args_and_kwargs(func: Callable, *args, **kwargs)->dict[str, Any]:
    '''unpack the args and kwargs to a dict'''
    sig = signature(func)
    params = sig.parameters
    params = {k: v for k, v in params.items()}
    params.pop('self', None)
    params.pop('cls', None)
    
    param_dict = {}
    
    for i, (param_name, param) in enumerate(params.items()):
        if param.kind == Parameter.VAR_POSITIONAL:
            param_dict[param_name] = args[i:]
        elif param.kind == Parameter.VAR_KEYWORD:
            param_dict[param_name] = kwargs
        elif i < len(args):
            param_dict[param_name] = args[i]
        elif param_name in kwargs:
            param_dict[param_name] = kwargs[param_name]
        else:
            param_dict[param_name] = param.default
    param_dict.update(kwargs)
    return param_dict

def get_required_params(func: Callable)->list[str]:
    '''Get the required params of a function excluding `self` and `cls`'''
    sig = signature(func)
    params = sig.parameters
    params = {k: v for k, v in params.items()}
    params.pop('self', None)
    params.pop('cls', None)
    
    required_params = []
    for param_name, param in params.items():
        if param.default == _empty:
            required_params.append(param_name)
    return required_params

def check_current_frame_is_under_class()->bool:
    '''
    check whether the current frame is under a class,
    e.g.
    ```python
    def wrapper(f):
        print(check_current_frame_is_under_class())

    class A:
        @wrapper    # True, cuz under class `A`
        @classmethod
        def method(cls, x):
            print(check_current_frame_is_under_class()) # False! since calling stack under A != frame under A
    '''
    frame = inspect.currentframe()
    if frame:
        frame = frame.f_back    # skip this function
    
    while frame:
        if frame.f_code.co_name == "<module>":
            return False
        f_locals = frame.f_locals
        if (qualname:=f_locals.get('__qualname__', None)) and '__module__' in f_locals:
            if isinstance(qualname, str):
                if qualname.split('.')[-1] == frame.f_code.co_name:
                    return True
        frame = frame.f_back
    return False
    

__all__ = [
    'pack_function_params', 
    'func_param_type_check', 
    'is_empty_method', 
    'func_arg_count', 
    'get_func_name', 
    'unpack_args_and_kwargs', 
    'get_required_params',
    'check_current_frame_is_under_class',
]


if __name__ == '__main__':
    from functools import partial
    
    def check_empty_func():
        class A:
            def t1(self):...
            
            def t2(self, x):
                '''hi'''
                # testing
                return None
                return
            
            def t3(self, x):
                '''hi
                hi
                hi'''
                raise NotImplementedError
                raise NotImplementedError('hi')    
            
            def test_empty(self):
                print(is_empty_method(self.t1)) # True
                print(is_empty_method(self.t2)) # True 
                print(is_empty_method(self.t3)) # True
                print(is_empty_method(self.test_empty)) # False
                
        a = A()
        a.test_empty()
        
    def check_frame_under_cls():
        class A:
            @classmethod
            def Wrapper(cls, f):
                print(check_current_frame_is_under_class(), 'is True')
                return f
            
        class B:
            @A.Wrapper
            @classmethod
            def method(cls):
                print(check_current_frame_is_under_class(), 'is False')
        
        B.method()
    
    def check_pack_param():
        def f(a, *b, **c):
            pass
        print(pack_function_params(f, ('a', 'b1', 'b2'), {'c1': 1, 'c2': 2}))
        
    # check_empty_func()
    # check_frame_under_cls()
    check_pack_param()