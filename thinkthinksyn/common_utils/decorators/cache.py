# -*- coding: utf-8 -*-
"""Decorators for caching functions/properties."""

from weakref import ref
from _thread import RLock
from functools import lru_cache, wraps, update_wrapper, _CacheInfo, _make_key
from typing import (Literal, overload, Callable, Hashable, TYPE_CHECKING, no_type_check, TypeAlias, TypeVar)
from typing_extensions import override

if TYPE_CHECKING:
    from .classproperty import class_property
    from ..concurrent_utils import AsyncFuncType

def _is_serializable(obj) -> bool:
    # for avoiding circular import
    from ..type_utils import is_serializable as _is_serializable_inner
    return _is_serializable_inner(obj)

# modify from functools.lru_cache to support async func
def _async_lru_cache_wrapper(user_function, maxsize, typed):
    sentinel = object()
    make_key = _make_key
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3
    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get
    cache_len = cache.__len__
    lock = RLock()
    root = []
    root[:] = [root, root, None, None]

    if maxsize == 0:

        async def wrapper(*args, **kwds):
            nonlocal misses
            misses += 1
            result = await user_function(*args, **kwds)
            return result

    elif maxsize is None:

        async def wrapper(*args, **kwds):
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = await user_function(*args, **kwds)
            cache[key] = result
            return result

    else:

        async def wrapper(*args, **kwds):
            nonlocal root, hits, misses, full
            key = make_key(args, kwds, typed)
            with lock:
                link = cache_get(key)
                if link is not None:
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result
                misses += 1
            result = await user_function(*args, **kwds)
            with lock:
                if key in cache:
                    pass
                elif full:
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    root[KEY] = root[RESULT] = None
                    del cache[oldkey]
                    cache[key] = oldroot
                else:
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    full = cache_len() >= maxsize
            return result

    def cache_info():
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper


class _CachedFunc:

    def __init__(self, func, maxsize=128, typed=False):
        if not (callable(func) or isinstance(func, property) or isinstance(func, (classmethod, staticmethod))):
            raise TypeError(f"Expected a function or property, but got {func}({type})")
        is_property = False
        binding_type: Literal["ins", "cls", "none"] = "none"
        if isinstance(func, property):
            from .classproperty import class_property

            if isinstance(func, class_property):
                binding_type = "cls"
            else:
                binding_type = "ins"
            is_property = True
            func = func.fget or func.getter  # type: ignore
        elif isinstance(func, classmethod):
            binding_type = "cls"
            func = func.__func__
        elif isinstance(func, staticmethod):
            binding_type = "none"
            func = func.__func__

        self.lru_cache_args = (maxsize, typed)
        self.methods = {}  # built methods
        self.origin_func = func
        self.is_property = is_property
        self.binding_type = binding_type
        update_wrapper(self, self.origin_func)  # type: ignore

    def __set_name__(self, owner, name):
        if self.binding_type == "none":
            self.binding_type = "ins"

    def _regist_cache(self, f, binder, is_property, lru_args):
        f = self._wrap_bound_cached_method(f, binder, is_property)  # type: ignore
        cache_f = lru_cache(*lru_args)(f)
        wraps(f)(cache_f)
        return cache_f

    @no_type_check
    def _wrap_bound_cached_method(self, func, binder, is_property):
        if binder is None:
            if is_property:
                def wrapper():  # type: ignore
                    return func()
            else:
                def wrapper(*args, **kwargs):  # type: ignore
                    return func(*args, **kwargs)
        else:
            if is_property:
                def wrapper():  # type: ignore
                    return func(binder())
            else:
                def wrapper(*args, **kwargs):  # type: ignore
                    return func(binder(), *args, **kwargs)
        return wrapper

    def __get__(self, instance, owner):
        if self.binding_type == "none":
            raise TypeError("Cannot bind a static method to an instance or class")
        binding = instance if self.binding_type == "ins" else owner
        binding_id = id(binding)
        if binding_id not in self.methods:
            _binder = ref(binding, lambda _: self.methods.pop(binding_id, None))
            weakly_bound_cached_method = self._regist_cache(
                self.origin_func, _binder, self.is_property, self.lru_cache_args
            )
            self.methods[binding_id] = weakly_bound_cached_method
        else:
            weakly_bound_cached_method = self.methods[binding_id]  # type: ignore
        if self.is_property:
            return weakly_bound_cached_method()
        return weakly_bound_cached_method

    def __call__(self, *args, **kwargs):
        if self.is_property:
            raise TypeError("Cannot call a property directly, use it as an attribute")
        if self.binding_type == "none":
            # check if all arguments are valid for caching
            for arg in args:
                if not isinstance(arg, Hashable) and not _is_serializable(arg):
                    return self.origin_func(*args, **kwargs)  # type: ignore
            for val in kwargs.values():
                if not isinstance(val, Hashable) and not _is_serializable(val):
                    return self.origin_func(*args, **kwargs)  # type: ignore

            # do like a normal @cache decorator
            if None not in self.methods:  # means not initialized
                method = self._regist_cache(self.origin_func, None, self.is_property, self.lru_cache_args)
                self.methods[None] = method
            return self.methods[None](*args, **kwargs)  # type: ignore
        else:
            if not args:
                raise TypeError("Expected at least one argument for a bound method")
            binding, args = args[0], args[1:]  # type: ignore
            binding_id = id(binding)
            if binding_id not in self.methods:
                _binder = ref(binding, lambda _: self.methods.pop(binding_id, None))
                weakly_bound_cached_method = self._regist_cache(
                    self.origin_func, _binder, self.is_property, self.lru_cache_args
                )
                self.methods[binding_id] = weakly_bound_cached_method
            else:
                weakly_bound_cached_method = self.methods[binding_id]  # type: ignore
            return weakly_bound_cached_method(*args, **kwargs)  # type: ignore

class _CacheAsyncFunc(_CachedFunc):
    @override
    def _regist_cache(self, f, binder, is_property, lru_args):
        f = self._wrap_bound_cached_method(f, binder, is_property)  # type: ignore
        cache_f = _async_lru_cache_wrapper(f, *lru_args)
        wraps(f)(cache_f)
        return cache_f


_CacheFuncTypes: TypeAlias = "Callable|property|class_property|classmethod|staticmethod"
_AsyncCacheFuncTypes: TypeAlias = "AsyncFuncType|property|class_property|classmethod|staticmethod"
_F = TypeVar('_F', bound=_CacheFuncTypes)
_AF = TypeVar('_AF', bound=_AsyncCacheFuncTypes)

@overload
def cache(func: _F, /) -> _F: ...
@overload
def cache(*, maxsize: int = 128, typed: bool = False) -> Callable[[_F], _F]: ...

def cache(*args, **kwargs):  # type: ignore
    """
    Cache any type of functions(normal/staticmethod/classmethod/instance method).
    Difference from `functools.cache`, this method supports caching non-hashable arguments.

    Args:
        maxsize: the max pool size of the cache(LRU)
        typed: whether to distinguish the type of the arguments, e.g. 1 and 1.0 are different when typed==True

    e.g.:
    ```
    @cache(maxsize=128, typed=False)    # can add arguments to the decorator
    def test(x):                        # normal function can be cached
        time.sleep(1)
        print('test', x)
        return x

    class A:
        @cache
        def f(self, x):
            print('f', x)
            time.sleep(1)
            return x

        @cache
        @classmethod
        def g(cls, x):
            print('g', x)
            time.sleep(1)
            return x

        @cache
        @staticmethod
        def h(x):
            print('h', x)
            time.sleep(1)
            return x

        @cache
        @property
        def i(self):
            print('i')
            time.sleep(1)
            return 1

        @cache
        @class_property
        def j(cls):
            print('j')
            time.sleep(1)
            return 1
    ```
    """
    if len(args) == 1 and (
        callable(args[0]) or isinstance(args[0], (classmethod, staticmethod)) or isinstance(args[0], property)
    ):
        return _CachedFunc(args[0])
    else:
        real_args = {"maxsize": 128, "typed": False}
        if args:
            real_args["maxsize"] = args[0]
            if len(args) > 1:
                real_args["typed"] = args[1]  # type: ignore
        real_args.update(kwargs)
        return lambda func: _CachedFunc(func, **real_args)  # type: ignore


@overload
def async_cache(func: _AF, /) -> _AF: ...
@overload
def async_cache(*, maxsize: int = 128, typed: bool = False) -> Callable[[_AF], _AF]: ...

def async_cache(*args, **kwargs):  # type: ignore
    """
    Async version of cache decorator, supports async functions.

    Args:
        maxsize: the max pool size of the cache(LRU)
        typed: whether to distinguish the type of the arguments, e.g. 1 and 1.0 are different when typed==True

    e.g.:
    ```
    @async_cache(maxsize=128, typed=False)    # can add arguments to the decorator
    async def test(x):                        # async function can be cached
        await asyncio.sleep(1)
        print('test', x)
        return x

    class A:
        @async_cache
        async def f(self, x):
            print('f', x)
            await asyncio.sleep(1)
            return x

        @async_cache
        @classmethod
        async def g(cls, x):
            print('g', x)
            await asyncio.sleep(1)
            return x

        @async_cache
        @staticmethod
        async def h(x):
            print('h', x)
            await asyncio.sleep(1)
            return x

        @async_cache
        @property
        async def i(self):
            print('i')
            await asyncio.sleep(1)
            return 1

        @async_cache
        @class_property
        async def j(cls):
            print('j')
            await asyncio.sleep(1)
            return 1
    ```
    """
    if len(args) == 1 and (
        callable(args[0]) or isinstance(args[0], (classmethod, staticmethod)) or isinstance(args[0], property)
    ):
        return _CacheAsyncFunc(args[0])
    else:
        real_args = {"maxsize": 128, "typed": False}
        if args:
            real_args["maxsize"] = args[0]
            if len(args) > 1:
                real_args["typed"] = args[1]  # type: ignore
        real_args.update(kwargs)
        return lambda func: _CacheAsyncFunc(func, **real_args)  # type: ignore


__all__ = ["cache", "async_cache"]


if __name__ == "__main__":  # debug
    import time, asyncio
    from .classproperty import class_property

    def test_cache():
        class A:
            @cache
            @property
            def f(self):
                print("a.f")
                time.sleep(1)
                return 1

            @cache  # type: ignore
            @class_property
            def g(cls):
                print("a.g")
                time.sleep(1)
                return 1

            @cache(maxsize=2)
            @classmethod
            def F(cls, x):
                print("a.F", x)
                time.sleep(1)
                return x

        print(A.F(1))
        print(A.F(1))

        print(A.g)
        print(A.g)
        a = A()
        print(a.f)
        print(a.f)

        class B:
            @cache(maxsize=2)
            def f(self, x):
                print(self, "b.f", x)
                time.sleep(1)
                return x

        b = B()
        b2 = B()
        b.f(a)
        b.f(a)
        b.f(1)
        b2.f(a)
        del b2
        b.f(1)

    async def test_async_cache():
        class A:
            @async_cache
            async def f(self, x):
                print("a.f", x)
                await asyncio.sleep(1)
                return x

            @async_cache  # type: ignore
            @class_property  # type: ignore
            async def g(cls):
                print("a.g")
                await asyncio.sleep(1)
                return 1

            @async_cache(maxsize=2)
            @classmethod
            async def F(cls, x):
                print("a.F", x)
                await asyncio.sleep(1)
                return x

        print(await A.F(1))
        print(await A.F(1))

        print(await A.g)
        print(await A.g)
        a = A()
        print(await a.f(1))
        print(await a.f(1))

    asyncio.run(test_async_cache())
