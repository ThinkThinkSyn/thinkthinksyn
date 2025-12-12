import logging
import multiprocessing.process as process_module
import multiprocessing.managers as manager_module

from pickle import PickleError
from types import FunctionType
from weakref import WeakValueDictionary
from multiprocessing.managers import BaseManager, BaseProxy
from typing import Self, ClassVar, TYPE_CHECKING, Any, Generator, Callable, TypeVar
from typing_extensions import override

from .file_lock import FileCrossProcessLock

_logger = logging.getLogger(__name__)

class _Manager(BaseManager):
    _is_creator: bool = False

    def __init__(self, address=None, authkey=None, serializer='pickle',
                 ctx=None, *, shutdown_timeout=1.0):
        super().__init__(address, authkey, serializer, ctx, shutdown_timeout=shutdown_timeout)

    @property
    def _lock_name(self):
        prefix = self.__class__.__name__
        if self.address:
            if isinstance(self.address, tuple) and len(self.address) == 2:
                return f'{prefix}.{self.address[0]}:{self.address[1]}'
            return f'{prefix}.{self.address}'
        return prefix

    def start(self):
        with FileCrossProcessLock(self._lock_name):
            try:
                self.connect()
            except FileNotFoundError:
                self._is_creator = True
                super().start()
    
    def __del__(self):
        if self._is_creator:
            try:
                self.shutdown()
            except Exception:
                pass

    @classmethod
    def register(cls, typeid, model, proxytype, exposed=None,
                 method_to_typeid=None, create_method=True):
        if '_registry' not in cls.__dict__:
            cls._registry = cls._registry.copy()

        exposed = exposed or getattr(proxytype, '_exposed_', None)
        method_to_typeid = method_to_typeid or \
                           getattr(proxytype, '_method_to_typeid_', None)

        if method_to_typeid:
            for key, value in list(method_to_typeid.items()):
                assert type(key) is str, '%r is not a string' % key
                assert type(value) is str, '%r is not a string' % value

        cls._registry[typeid] = (model, exposed, method_to_typeid, proxytype)

        if create_method:
            def temp(self, /, *args, **kwds):
                token, exp = self._create(typeid, *args, **kwds)
                proxy = proxytype(
                    model, args[0], token, self._serializer, manager=self,  # type: ignore
                    authkey=self._authkey, exposed=exp
                    )
                conn = self._Client(token.address, authkey=self._authkey)   
                manager_module.dispatch(conn, None, 'decref', (token.id,))  # type: ignore
                return proxy
            temp.__name__ = typeid
            setattr(cls, typeid, temp)

_proxy_type_cache = {}
_empty_init = object.__init__
_existing_proxy_attrs = set(
    ('_tls', '_idset', '_token', '_id', '_manager', '_serializer', '_Client', '_owned_by_manager', 
     '_authkey', 'shared_obj_id', '_get_manager', '_get_unknown_attribute') + tuple(dir(BaseProxy))
)
_R = TypeVar('_R')

class _BaseProxy(BaseProxy):

    _shared_obj_id: str
    _on_method_call: Callable[[str, tuple, dict[str, Any]], Generator[tuple[tuple, dict[str, Any]], Any, None]]|None = None
    _on_value_gotten: Callable[[str, Any], Any]|None = None
    
    @property
    def shared_obj_id(self)->str:
        return getattr(self, '_shared_obj_id')

    @override
    def _callmethod(self, methodname: str, args: tuple=tuple(), kwds: dict[str, Any]={}):
        method_wrapper = None
        if self._on_method_call is not None:
            method_wrapper = self._on_method_call(methodname, args, kwds)
            args, kwds = next(method_wrapper)
        r = super()._callmethod(methodname, args, kwds)
        if method_wrapper is not None:
            try:
                r = method_wrapper.send(r)
            except StopIteration:
                pass
        return r

    def __getattr__(self, name):
        if not name.startswith('_') and (not name in _existing_proxy_attrs):
            try:
                r = self._callmethod('_get_unknown_attribute', (name,))
                if self._on_value_gotten is not None:
                    r = self._on_value_gotten(name, r)
                return r
            except PickleError as e:
                raise AttributeError(f'Attribute {name} is not serializable and cannot be retrieved from manager process.') from e
        raise AttributeError(f'Attribute {name} not found.')

def _make_proxy_type(name, exposed):
    exposed = tuple(exposed)
    try:
        return _proxy_type_cache[(name, exposed)]
    except KeyError:
        pass

    dic = {}
    for meth in exposed:
        exec('''def %s(self, /, *args, **kwds):
        return self._callmethod(%r, args, kwds)''' % (meth, meth), dic)
    
    ProxyType = type(name, (_BaseProxy,), dic)
    ProxyType._exposed_ = exposed
    _proxy_type_cache[(name, exposed)] = ProxyType
    return ProxyType

def _auto_proxy(cls: type['CrossProcessSharedObject'], id, token, serializer, manager=None, authkey=None,
              exposed=None, incref=True, manager_owned=False):
    if (p:=cls.__Proxies__.get(id, None)) is not None:
        return p
    _Client = manager_module.listener_client[serializer][1] # type: ignore

    if exposed is None:
        conn = _Client(token.address, authkey=authkey)
        try:
            exposed = manager_module.dispatch(conn, None, 'get_methods', (token,))  # type: ignore
        finally:
            conn.close()

    if authkey is None and manager is not None:
        authkey = manager._authkey
    if authkey is None:
        authkey = process_module.current_process().authkey
    
    ProxyType = _make_proxy_type(f'{cls.__name__}Proxy', exposed)
    proxy = ProxyType(token, serializer, manager=manager, authkey=authkey,
                      incref=incref, manager_owned=manager_owned)
    # set proxy attributes
    proxy._isauto = True    # type: ignore
    proxy._shared_obj_id = id  # type: ignore
    if hasattr(cls, '__on_method_call__'):
        on_method_call = getattr(cls, '__on_method_call__')
        if callable(on_method_call):
            proxy._on_method_call = on_method_call  # type: ignore
    if hasattr(cls, '__on_value_gotten__'):
        on_value_gotten = getattr(cls, '__on_value_gotten__')
        if callable(on_value_gotten):
            proxy._on_value_gotten = on_value_gotten  # type: ignore

    cls.__Proxies__[id] = proxy  # type: ignore
    return proxy

class CrossProcessSharedObject:
    '''
    `CrossProcessSharedObject` is a base class for creating shared objects that can be accessed
    across multiple processes through `multiprocessing.Manager`.

    Shared objects are created on an independent process, and when a method/value is accessed, 
    proxy forward the call to the manager process and get the result back(dump/load using `pickle`).
    
    Shared objects are identified by a unique `id` string. When creating a shared object with the same `id`,
    the existing instance will be returned instead of creating a new one. Note that the object will
    not initialize again when the same `id` is used, even different initialization arguments are provided.

    All custom methods which is NOT starting with '_' are automatically exposed to the proxy.
    Similarly, all custom attributes which is NOT starting with '_' can be accessed through the proxy.
    Note that only picklable values can be accessed.

    You can also define `__on_method_call__` and `__on_value_gotten__` class methods to customize the behavior
    of method calls and value retrievals on the proxy side.

    Here shows an example:
    ```
    class A(CrossProcessSharedObject):
        def __init__(self, id: str, /, x:int, y:int):
            # you dont need to deal with `id`.
            self.x = x
            self.y = y
            self.count = 0
    
        def foo(self, z: int)->int:
            # NOTE: here, for simple demo, no thread locks are used for `self.count`
            #       in real case, you should better use thread lock to ensure atomic operations.
            self.count += 1
            return self.x + self.y + z
        
        def bar(self, x)->int:
            return self.x + x

        @classmethod
        def __on_method_call__(cls, name: str, args: tuple, kwargs: dict[str, Any]):
            if name == 'f':
                x = args[0] if args else kwargs.get('x', 0)
                r = yield (tuple(), {'x': x + 1})  # modify argument x
                yield r * 2  # modify return value
            else:
                r = yield (args, kwargs)
                yield r

    def worker():
        a = A('shared_a', x=1, y=2)
        print(f'({os.getpid()}) a.foo(3)={a.foo(3)}, count={a.count}')  # will trigger `foo` in manager process
        print(f'({os.getpid()}) a.bar(1)={a.bar(1)}')   # will output 6, since (1 + 1 + 1)*2 == 6

    with ProcessPoolExecutor(5) as executor:
        futures = [executor.submit(worker) for _ in range(5)]
        for future in futures:
            future.result()
    ```
    '''

    __ManagerClass__: ClassVar[type[_Manager]]              # used in current process only
    __Manager__: ClassVar[_Manager]                         # used in current process only
    __Proxies__: ClassVar[WeakValueDictionary[str, Self]]     # used in current process only
    __Instances__: ClassVar[WeakValueDictionary[str, Self]]   # used in manager process only
    __InManagerProcess__: bool = False
    __OriginInit__: FunctionType
    
    if not TYPE_CHECKING:
        def __new__(cls, id: str|None=None, /, **kwargs):
            if not id:
                id = cls.__name__
            if cls.__InManagerProcess__:
                lock_key = f'{cls.__name__}_instance_lock_{id}'
                with FileCrossProcessLock(lock_key):
                    if (ins:=cls.__Instances__.get(id, None)) is not None:
                        return ins
                    _logger.debug(f'Creating CrossProcessSharedObject of `{cls.__name__}` in manager process, id=`{id}`.')
                    ins = super().__new__(cls)
                    setattr(ins, '_shared_obj_id', id)
                    cls.__Instances__[id] = ins
                    if cls.__OriginInit__ is not _empty_init:
                        cls.__OriginInit__(ins, id, **kwargs)
                    return ins
            else:
                if (p:=cls.__Proxies__.get(id, None)) is not None:
                    return p
                manager = cls._get_manager()
                proxy = manager.__getattribute__(cls.__name__)(id, **kwargs)
                # `__Proxies__[id]` is updated in `_auto_proxy`
                return proxy
            
        def __init_subclass__(cls, is_subprocess=False) -> None:
            setattr(cls, '__InManagerProcess__', is_subprocess)
            origin_init = getattr(cls, '__init__', _empty_init)
            if origin_init is _empty_init and len(cls.__mro__) >2:
                parent_cls = cls.__mro__[1]
                parent_origin_init = getattr(parent_cls, '__OriginInit__', None)
                if parent_origin_init not in (_empty_init, None):
                    origin_init = parent_origin_init
            setattr(cls, '__OriginInit__', origin_init)
            setattr(cls, '__init__', _empty_init)
            
            if not is_subprocess:
                manager_cls = type(
                    f'{cls.__name__}Manager',
                    (_Manager,),
                    {}
                )
                setattr(cls, '__ManagerClass__', manager_cls)
                setattr(cls, '__Proxies__', WeakValueDictionary())

                exposes = set(['_get_unknown_attribute',])
                for attr_name in dir(cls):
                    if not attr_name.startswith('_') and not hasattr(CrossProcessSharedObject, attr_name):
                        attr = getattr(cls, attr_name)
                        if isinstance(attr, FunctionType) and not isinstance(attr, staticmethod):
                            exposes.add(attr_name)
                _subprocess_cls = type(
                    cls.__name__,
                    (cls,),
                    {},
                    is_subprocess=True
                )
                manager_cls.register(cls.__name__, _subprocess_cls, proxytype=_auto_proxy, exposed=tuple(exposes))
            else:
                setattr(cls, '__Instances__', WeakValueDictionary())
         
        @classmethod
        def _get_manager(cls) -> _Manager:
            if (m:=getattr(cls, '__Manager__', None)) is None:
                m = cls.__ManagerClass__(cls.__name__)
                m.start()
                setattr(cls, '__Manager__', m)
            return m    # type: ignore
        
        def _get_unknown_attribute(self, name: str):
            return getattr(self, name)
        
    else:
        def __init__(self, id: str|None=None, /, **kwargs):
            '''
            Create or get a shared object instance with the given `id`.
            If left empty, id = `cls.__name__`.
            
            `**kwargs` are for your custom initialization parameters.
            '''
    
    @property
    def shared_obj_id(self) -> str:
        return getattr(self, '_shared_obj_id')  # type: ignore
    
    if TYPE_CHECKING:
        @classmethod
        def __on_method_call__(cls, name: str, args: tuple, kwargs: dict)->Generator[tuple[tuple, dict[str, Any]], Any, None]: 
            '''
            If this method is defined, it will be called on client side(i.e. proxy side) before
            it forward the to manager process. The method should be a generator that yields the (args, kwargs) 
            tuple to be used for the actual method call in manager process, and can receive the return value 
            from manager process after the method call.

            NOTE: this method must be static or class method.
            
            Example:
            ```python
            class A(CrossProcessSharedObject):
                def f(self, x: int):
                    return x + 1
                
                @classmethod
                def __on_method_call__(cls, name: str, args: tuple, kwargs: dict[str, Any]):
                    if name == 'f':
                        x = args[0] if args else kwargs.get('x', 0)
                        r = yield (tuple(), {'x': x + 1})  # modify argument x
                        yield r * 2  # modify return value
                    else:
                        r = yield (args, kwargs)
                        yield r
            ```
            '''
            ...

        @classmethod
        def __on_value_gotten__(cls, name: str, value: _R)->_R:
            '''
            If this method is defined, it will be called on client side(i.e. proxy side) after
            it gets the return value from manager process. You can use this method to modify the return value.

            NOTE: this method must be static or class method.
            '''
            ...

    

__all__ = ['CrossProcessSharedObject']


if __name__ == "__main__":
    import os
    from concurrent.futures import ProcessPoolExecutor

    class A(CrossProcessSharedObject):
        def __init__(self, id: str, /, x:int, y:int):
            self.x = x
            self.y = y
            self.count = 0
    
        def foo(self, z: int)->int:
            self.count += 1
            return self.x + self.y + z
        
        def bar(self, x)->int:
            return self.x + x
        
        @property
        def x_plus_y(self):
            return self.x + self.y

        @classmethod
        def __on_method_call__(cls, name: str, args: tuple, kwargs: dict[str, Any]):
            if name == 'bar':
                x = args[0] if args else kwargs.get('x', 0)
                r = yield (tuple(), {'x': x + 1})  # modify argument x
                yield r * 2  # modify return value
            else:
                r = yield (args, kwargs)
                yield r

    def worker():
        a = A('shared_a', x=1, y=2)
        print(f'({os.getpid()}) a.foo(3)={a.foo(3)}, count={a.count}')
        print(f'({os.getpid()}) a.bar(1)={a.bar(1)} == (1 + 1 + 1)*2 == 6')
        print(f'({os.getpid()}) a.x_plus_y={a.x_plus_y}')

    with ProcessPoolExecutor(5) as executor:
        futures = [executor.submit(worker) for _ in range(5)]
        for future in futures:
            future.result()