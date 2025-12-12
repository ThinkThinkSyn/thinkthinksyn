# -*- coding: utf-8 -*-
'''Async related utils.'''
import asyncio
import traceback
import threading
import contextvars
import logging

from functools import wraps, partial
from asyncio import _get_running_loop
from multiprocessing import cpu_count
from types import FunctionType, MethodType
from concurrent.futures import ThreadPoolExecutor, Future
from typing import (Callable, TypeVar, Awaitable, TypeGuard, AsyncIterable, overload, 
                    AsyncGenerator, Any, ItemsView, Coroutine, Generator)
from typing_extensions import ParamSpec, TypeAliasType

from .nested_loop import get_nested_loop_policy

_R = TypeVar('_R')
_F = TypeVar('_F', bound=Callable[..., Any])
_P = ParamSpec('_P')

AsyncFuncType = TypeAliasType("AsyncFuncType", Callable[..., Awaitable[Any]])
'''Type hints for async functions'''
AsyncFunc = TypeAliasType("AsyncFunc", Callable[_P, Awaitable[_R]], type_params=(_P, _R))
'''Type hints for async functions with parameters'''
SyncOrAsyncFunc = TypeAliasType("SyncOrAsyncFunc", Callable[_P, Awaitable[_R]]|Callable[_P, _R], type_params=(_P, _R))
'''Type hints for sync or async functions'''
SyncOrAsyncGenerator = TypeAliasType("SyncOrAsyncGenerator", AsyncGenerator[_R, None]|Generator[_R, Any, None], type_params=(_R,))
'''Type hints for sync or async generators'''

_logger = logging.getLogger(__name__)

__background_threadpool__ = None

def get_threadpool()->ThreadPoolExecutor:
    '''Get the global thread pool for running background tasks'''
    global __background_threadpool__
    if __background_threadpool__ is None:
        __background_threadpool__ = ThreadPoolExecutor(max(cpu_count(), 1))
    return __background_threadpool__

def is_in_running_async_loop()->bool:
    '''check whether the code is within an *running* async loop'''
    loop = _get_running_loop()
    if not loop:
        return False
    return loop.is_running()

def is_in_main_thread()->bool:
    '''check whether the code is running in the main thread'''
    return threading.current_thread() == threading.main_thread()
    
def run_async(coro: Awaitable[_R], create_if_not_exist=True)->_R:
    '''
    Wrapper of `asyncio.run`, but this method will not create a new event loop if exists
    if `create_if_not_exist` is False. In that case, RuntimeError will be raised.
    '''
    if not create_if_not_exist:
        if not (loop:=_get_running_loop()):
            raise RuntimeError('No running event loop')
        return loop.run_until_complete(coro)
    else:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
def run_func_with_context(
    f: Callable[_P, _R], 
    context: contextvars.Context,
    *args: _P.args,
    **kwargs: _P.kwargs
) -> _R:
    '''
    Run a function with a specific context, e.g. for copying
    main thread's context to another thread.
    '''
    ctx = context.copy()
    return ctx.run(f, *args, **kwargs)

@overload
def run_in_background(func:Callable[..., Awaitable[_R]], args:tuple|None=None, kwargs: dict|None=None, timeout: int|None=120)->Future[_R]:...
@overload
def run_in_background(func:Callable[..., _R], args:tuple|None=None, kwargs: dict|None=None, timeout: int|None=120)->Future[_R]:...
@overload
def run_in_background(func:Callable[..., _R]|Callable[..., Awaitable[_R]], args:tuple|None=None, kwargs: dict|None=None, timeout: int|None=120)->Future[_R]:...

def run_in_background(func:Callable, args:tuple|None=None, kwargs: dict|None=None, timeout: int|None=120):  # type: ignore
    '''
    Run a function in background, i.e., submit the function to another thread to run.
    The function can be both sync and async.
    If `timeout` is set, the function will be cancelled after `timeout` seconds.
    '''
    args = args or tuple()
    kwargs = kwargs or dict()
    _pool = get_threadpool()
    
    if is_async_callable(func):
        def wrapper(f, context_items: ItemsView[contextvars.ContextVar, Any], *args, **kwargs):
            for context_var, value in context_items:
                context_var.set(value)
            loop = get_nested_loop_policy().get_or_create_event_loop()
            try:
                if timeout and timeout > 0: # type: ignore
                    coro = asyncio.wait_for(f(*args, **kwargs), timeout)
                else:
                    coro = f(*args, **kwargs)
                return loop.run_until_complete(coro)
            except asyncio.TimeoutError:
                _logger.error(f'(run_in_background) Async background function: {f} timeout after {timeout} seconds.')
            except Exception as e:
                _logger.error(f'(run_in_background) Error occur when running background function: {f}. Err={type(e).__name__}:{e}')

        ctx = contextvars.copy_context()
        return _pool.submit(wrapper, func, ctx.items(), *args, **kwargs)
    else:
        def wrapper(f, context_items: ItemsView[contextvars.ContextVar, Any], *args, **kwargs):
            for context_var, value in context_items:
                context_var.set(value)
            try:
                return f(*args, **kwargs)
            except Exception as e:
                _logger.error(f'(run_in_background) Error occur when running background function: {f}. Err={type(e).__name__}:{e}')
        
        ctx = contextvars.copy_context()
        f = _pool.submit(wrapper, func, ctx.items(), *args, **kwargs)
        if timeout and timeout > 0: # type: ignore
            def timeout_wrapper(f: Future, timeout: int):
                try:
                    return f.result(timeout=timeout)
                except TimeoutError:
                    _logger.error(f'(run_in_background) Sync background function: {f} timeout after {timeout} seconds.')
            return _pool.submit(timeout_wrapper, f, timeout)
        else:
            return f

def is_async_callable(func:Callable)->TypeGuard[Callable[..., Awaitable]]:
    '''
    Check if a callable is async.
    For special objects, you can define `__is_async_func__` method to return the async status.
    '''
    while isinstance(func, partial):
        func = func.func
    if isinstance(func, (staticmethod, classmethod)):
        func = func.__func__
        
    if not isinstance(func, (FunctionType, MethodType)):
        if hasattr(func, '__call__') and not type(func.__call__).__qualname__ == 'method-wrapper':
            func = func.__call__    # type: ignore
    elif isinstance(func, MethodType) and hasattr(func, '__self__'):
        func = func.__func__
    
    if hasattr(func, '__is_async_func__'):
        from ..type_utils import func_arg_count
        from .async_helpers import wait_coroutine
        custom_func = getattr(func, '__is_async_func__')
        if callable(custom_func) and func_arg_count(custom_func) == 0:
            r = custom_func()
            if isinstance(r, Coroutine):
                r = wait_coroutine(r)
            try:
                return bool(r)
            except:
                _logger.warning(f'(is_async_callable) Error occur when calling `__is_async_func__` method of {func}. Will be decided by default implementation.')
        else:
            _logger.warning(f'(is_async_callable) `__is_async_func__` method of {func} is not callable or has invalid argument count. Will be decided by default implementation.')
    return asyncio.iscoroutinefunction(func)

def to_timeout_func(f: _F, timeout:int=120)->_F:
    '''
    Wrap a function with a timeout.
    TimeoutError will be raised if the function is not finished within the timeout.
    '''
    @wraps(f)
    def wrapper(*args, **kwargs):
        return run_in_background(f, args, kwargs, timeout).result()
    return wrapper  # type: ignore

async def wait_for_condition(conditioner:Callable[[], bool], interval:float=0.1, timeout:float=10):
    '''Wait for a condition to be true. If timeout, raise TimeoutError'''
    time = 0
    while time < timeout:
        if conditioner():
            return
        await asyncio.sleep(interval)
        time += interval
    raise TimeoutError(f'Wait for conditioner: {conditioner} timeout after {timeout} seconds.')


class _InternalStopAsyncIteration(Exception):
    """A special stop exception that also returns the finished generator's key."""
    key: int
    '''key in generator dict'''
    def __init__(self, key):
        self.key = key

async def combine_async_generators(
    *generators: AsyncIterable[_R]|AsyncGenerator[_R, None],
    raise_err: bool = True
)->AsyncGenerator[tuple[int, _R], None]:
    '''
    combine multiple async iterations into one, returns a new async generator which 
    yields (index, origin_val).
    
    E.g. 
    ```python
    async def gen(x: int):
        for i in range(x):
            await asyncio.sleep(0.1)
            yield x
    
    async def main():
        combined_gen = combine_async_generators(gen(1), gen(2))
        async for index, v in combined_gen:
            # do something
            # index will be 0 or 1 and v will be 1 or 2 in this case
            ... 
    ```
    '''
    async def anext(key, gen):
        try:
            n = await gen.__anext__()
            return key, n
        except StopAsyncIteration:
            raise _InternalStopAsyncIteration(key)
        
    generator_dict = {i: gen for i, gen in enumerate(generators)}
    pending = {asyncio.create_task(anext(key, gen)) for key, gen in generator_dict.items()}
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for i in done:
            exception = i.exception()
            if isinstance(exception, Exception):
                if isinstance(exception, _InternalStopAsyncIteration):
                    generator_dict.pop(exception.key)
                else:
                    if raise_err:
                        raise exception
                    else:
                        except_traceback = traceback.format_tb(exception.__traceback__)
                        except_traceback = ''.join(except_traceback)
                        _logger.error(f'error occur when running async iterator inside combine_async_generators. Err msg: {exception}. traceback: {except_traceback}')
            else:
                key, val = i.result()
                pending.add(asyncio.create_task(anext(key, generator_dict[key])))
                yield key, val

async def collect_async_stream_output(async_iterable: AsyncIterable[_R]|AsyncGenerator[_R, None])->list[_R]:
    '''Collect the output of an async stream into a list.'''
    results = []
    async for result in async_iterable:
        results.append(result)
    return results

_on_program_exit_listeners: list[tuple[Callable, tuple, dict]] = []  # type: ignore

@overload
def add_on_program_exit_listener(func: Callable[_P, Any], *args: _P.args, **kwargs: _P.kwargs):...
@overload
def add_on_program_exit_listener(func: Callable[_P, Awaitable[Any]], *args: _P.args, **kwargs: _P.kwargs):...

def add_on_program_exit_listener(func: Callable, *args, **kwargs):
    '''
    Add a listener to be called when the program exits.
    
    NOTE: 
        1. Listener functions will be called in reverse order of adding.
        2. Listeners will be called BEFORE `atexit` listeners, but (almost) AFTER `threading._threading_atexits` listeners.
        3. please do not use threading related function inside the listener, as the threading module may be already shutdown.
    '''
    _on_program_exit_listeners.append((func, args, kwargs))

__program_exited__ = False

def _on_program_exit():
    global __program_exited__
    if __program_exited__ or not _on_program_exit_listeners:
        return
    __program_exited__ = True
 
    _logger.debug(f'Running {len(_on_program_exit_listeners)} program exit listeners...')
    async_jobs = []
    for func, args, kwargs in _on_program_exit_listeners:
        if is_async_callable(func):
            async_jobs.append((func, args, kwargs))
        else:
            try:
                _logger.debug(f'Running program exit listener: {func} with args={args}, kwargs={kwargs}')
                func(*args, **kwargs)
            except Exception as e:
                _logger.error(f'Error occur when running on program exit listener: {func}. Err={type(e).__name__}:{e}')
    if async_jobs:
        async def no_exp_wrapper(func, *args, **kwargs):
            try:
                await func(*args, **kwargs)
            except Exception as e:
                _logger.error(f'Error occur when running on program exit listener: {func}. Err={type(e).__name__}:{e}')
        
        async def _run_async_jobs(async_jobs):
            await asyncio.gather(*[no_exp_wrapper(func, *args, **kwargs) for func, args, kwargs in async_jobs])
        
        asyncio.run(_run_async_jobs(async_jobs))

threading._threading_atexits.append(_on_program_exit)   # type: ignore

def is_program_exited()->bool:
    '''Check if the program has exited.'''
    return get_global_value('__program_exited__', False)  # type: ignore


__all__ = [
    'AsyncFuncType',
    'AsyncFunc',
    'SyncOrAsyncFunc', 
    'SyncOrAsyncGenerator',
    'get_threadpool', 
    'is_in_running_async_loop', 
    'is_in_main_thread', 
    'run_async', 
    'is_async_callable', 
    'to_timeout_func',
    'wait_for_condition', 
    'combine_async_generators',
    'run_func_with_context', 
    'run_in_background', 
    'collect_async_stream_output',
    'add_on_program_exit_listener',
    'is_program_exited',
]


if __name__ == '__main__':
    def test_run_in_bg():
        async def f():
            for i in range(10):
                await asyncio.sleep(0.3)
                print('f', i)
                
        def g():
            import time
            for i in range(10):
                time.sleep(0.3)
                print('g', i)
        
        def test_bg():
            run_in_background(f)
            run_in_background(g)
            
        test_bg()
        print('done')
    
    def test_is_async_callable():
        async def f():
            ...
        print(is_async_callable(f))
        print(is_async_callable(classmethod(f)))    # type: ignore
        print(is_async_callable(staticmethod(f)))   # type: ignore
        
    