# -*- coding: utf-8 -*-
'''Async related utils.'''
import os
import time
import atexit
import asyncio
import logging
import contextvars

from functools import wraps
from threading import Thread, Event
from dataclasses import dataclass, field
from queue import Queue, Empty as QueueEmpty
from typing import (Awaitable, Callable, Any, Iterable, overload, Generator, AsyncGenerator,
                    Iterable, AsyncIterable, Coroutine)
from typing_extensions import TypeVar, ParamSpec, TypeAliasType

from ..type_utils import check_value_is, Empty
from .nested_loop import get_nested_loop_policy
from .helper_funcs import AsyncFuncType, is_async_callable, SyncOrAsyncFunc

_logger = logging.getLogger(__name__)

def _is_uvloop(loop):
    return 'uvloop' in str(type(loop)).lower()

_R = TypeVar('_R')
_P = ParamSpec('_P')

def async_run(coro: Coroutine[Any, Any, _R], timeout: int|None=None) -> _R:
    '''
    Advanced version of `asyncio.run` to support nested loop.
    If the current loop is a nested loop, it will run the coroutine directly.
    If `timeout` is not None, it will be passed to `asyncio.wait_for`.
    
    NOTE: it will raise `RuntimeError` if there is running non-nested loop.
    '''
    if timeout is not None and timeout > 0:
        coro = asyncio.wait_for(coro, timeout)
    if (curr_loop := asyncio._get_running_loop()):
        if getattr(curr_loop, "_nest_patched", False) and not _is_uvloop(curr_loop):
            return curr_loop.run_until_complete(coro) # type: ignore
    return asyncio.run(coro)

def wait_coroutine(coro: Coroutine[Any, Any, _R], timeout: int|None=None) -> _R:
    '''
    Wait for a coroutine to finish and return the result.
    If `timeout` is not None, it will be passed to `asyncio.wait_for`.
    '''
    if timeout is not None and timeout > 0:
        coro = asyncio.wait_for(coro, timeout)
    if (curr_loop := asyncio._get_running_loop()):
        if getattr(curr_loop, "_nest_patched", False) and not _is_uvloop(curr_loop):
            return curr_loop.run_until_complete(coro) # type: ignore
    async def run(coro):
        return await coro
    return run_async_in_sync(run, coro) # type: ignore

@dataclass
class _AsyncTask:
    func: AsyncFuncType
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    context: contextvars.Context = field(default_factory=contextvars.copy_context)
    result: Any = Empty
    error: BaseException|Empty = Empty
    
    @property
    def finished(self)->bool:
        return (self.result is not Empty) or (self.error is not Empty)
    
    async def run(self):
        if self.finished:
            return
        try:
            r = self.context.run(self.func, *self.args, **self.kwargs)
            self.result = await r
            _logger.debug(f'async task `{self.func}` finished with result: {self.result}')
        except BaseException as e:
            _logger.warning(f'async task `{self.func}` raised exception. {type(e)}: {e}')
            self.error = e

__async_runner_thread_stop_event__ = None
__async_runner_task_queue__ = None
__async_runner_threads__ = {} # pid -> Thread

def _stop_async_runner_thread_event()->Event:
    global __async_runner_thread_stop_event__
    if __async_runner_thread_stop_event__ is None:
        __async_runner_thread_stop_event__ = Event()
    return __async_runner_thread_stop_event__

def _async_runner_task_queue()->Queue[_AsyncTask]:
    global __async_runner_task_queue__
    if __async_runner_task_queue__ is None:
        __async_runner_task_queue__ = Queue()
    return __async_runner_task_queue__

async def _run_async_in_thread():
    task_queue = _async_runner_task_queue()
    stop_event = _stop_async_runner_thread_event()
    while not stop_event.is_set():
        try:
            task = task_queue.get(block=False)
            asyncio.create_task(task.run())
            await asyncio.sleep(0)
        except QueueEmpty:
            await asyncio.sleep(0.01)

def _async_task_runner():
    loop = get_nested_loop_policy().get_or_create_event_loop()
    loop.run_until_complete(_run_async_in_thread())

def _start_async_runner_thread():
    key = f'__{os.getpid()}_async_runner_thread_initiated__'
    if key not in __async_runner_threads__:
        atexit.register(lambda: _stop_async_runner_thread_event().set())
        _async_runner_thread = Thread(target=_async_task_runner, daemon=True)
        __async_runner_threads__[key] = _async_runner_thread
        _async_runner_thread.start()
        
def run_async_in_sync(async_func: Callable[_P, Awaitable[_R]], *args: _P.args, **kwargs:_P.kwargs)->_R:
    '''
    Get the return value of an async function. The simple version of `run_async_funcs`.
    Note: async function will be submitted to another thread,
         thus please ensure the function is thread-safe.
    '''
    if (curr_loop := asyncio._get_running_loop()):
        if getattr(curr_loop, "_nest_patched", False) and not _is_uvloop(curr_loop):
            return async_run(async_func(*args, **kwargs)) # type: ignore
    
    _start_async_runner_thread()
    task = _AsyncTask(func=async_func, args=args, kwargs=kwargs)
    task_queue = _async_runner_task_queue()
    task_queue.put(task)
    while not task.finished:
        time.sleep(0.01)
    if task.error is not Empty:
        raise task.error    # type: ignore
    return task.result  # type: ignore

_AsyncFuncType = TypeAliasType("_AsyncFuncType", Callable[..., Awaitable[_R]], type_params=(_R,))

@overload
def run_async_funcs(
    async_funcs:_AsyncFuncType[_R]|Iterable[_AsyncFuncType[_R]], 
    args:tuple[tuple, ...]|None=None, 
    kwargs:tuple[dict]|dict|None=None,
    timeout: int|None=None
)->list[_R]: ...

@overload
def run_async_funcs(
    async_funcs:AsyncFuncType|Iterable[AsyncFuncType], 
    args:tuple[tuple, ...]|None=None, 
    kwargs:tuple[dict]|dict|None=None,
    timeout: int|None=None
)->list[Any]: ...
    
def run_async_funcs(
    async_funcs:AsyncFuncType|Iterable[AsyncFuncType], 
    args:tuple[tuple, ...]|None=None, 
    kwargs:tuple[dict]|dict|None=None,
    timeout: int|None=None
): # type: ignore
    '''
    Run async functions and get return. 
    
    This function is the multi version of `get_async_func_return`, i.e., it can run multiple 
    async functions at the same time, but args/kwargs should be passed in the same
    length as async_funcs.
    '''
    if not isinstance(async_funcs, Iterable):
        async_funcs = [async_funcs]
    if args is None:
        args = [tuple()] * len(async_funcs) # type: ignore
    elif not check_value_is(args, tuple[tuple]):    # only 1 tuple in outer tuple only, e.g. ((..), )
        args = [args,] * len(async_funcs) # type: ignore
    if kwargs is None:
        kwargs = [dict()] * len(async_funcs) # type: ignore
    elif kwargs is not None and isinstance(kwargs, dict):
        kwargs = [kwargs,] * len(async_funcs) # type: ignore
    
    if not timeout or timeout<=0:
        async def run():
            return await asyncio.gather(*[async_func(*arg, **kwarg) for async_func, arg, kwarg in zip(async_funcs, args, kwargs)]) # type: ignore
        return run_async_in_sync(run)
    else:
        async def run():
            f = asyncio.gather(*[async_func(*arg, **kwarg) for async_func, arg, kwarg in zip(async_funcs, args, kwargs)]) # type: ignore
            await asyncio.wait_for(f, timeout)
            return f.result()
        return run_async_in_sync(run)

@overload
def run_any_func(func: SyncOrAsyncFunc[_P, _R], *args: _P.args, **kwargs: _P.kwargs)->_R:...
@overload
def run_any_func(func: Callable[_P, Awaitable[_R]], *args: _P.args, **kwargs: _P.kwargs)->_R:...
@overload
def run_any_func(func: Callable[_P, _R],  *args: _P.args, **kwargs: _P.kwargs)->_R:...

def run_any_func(func, *args, **kwargs):
    '''
    Wrapper of `run_async_in_sync`, i.e. detect the function type,
    and call `run_async_in_sync` in case of async function.
    
    Note: for async function, it will be submitted to another thread to run,
         thus it can be a solution to run async function in parallel.
    '''
    if is_async_callable(func):
        return run_async_in_sync(func, *args, **kwargs)
    else:
        return func(*args, **kwargs)    # type: ignore


def run_any_func_with_timeout(
    func: Callable[..., _R],
    args: tuple|None = None,
    kwargs: dict|None = None,
    timeout: int|float|None = None,
)->_R:   # type: ignore
    '''
    Advanced version of `run_any_func` with timeout feature.
    When `timeout` is None, it is equivalent to `run_any_func`.
    '''
    args = args or tuple()
    kwargs = kwargs or {}
    
    if not timeout or timeout<=0:
        return run_any_func(func, *args, **kwargs)  # type: ignore
    if is_async_callable(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout)
        return run_async_in_sync(wrapper, *args, **kwargs)  # type: ignore
    else:
        from .helper_funcs import get_threadpool, run_func_with_context
        pool = get_threadpool()
        return pool.submit(
            run_func_with_context,
            func,
            contextvars.copy_context(),  # type: ignore 
            *args, 
            **kwargs
        ).result(timeout)   # type: ignore
    

@overload
async def async_run_any_func(func: SyncOrAsyncFunc[_P, _R], *args: _P.args, **kwargs: _P.kwargs)->_R:...
@overload
async def async_run_any_func(func: Callable[_P, Awaitable[_R]], *args: _P.args, **kwargs: _P.kwargs)->_R:...
@overload
async def async_run_any_func(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs)->_R:...

async def async_run_any_func(func, *args, **kwargs):
    '''
    async version of `run_any_func`.
     
    If the function is async, `await` will be called automatically, otherwise, 
    the function will be called directly.
    '''
    if is_async_callable(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)

@overload
async def async_wrap_sync_func(func: SyncOrAsyncFunc[_P, _R], *args: _P.args, **kwargs: _P.kwargs)->_R:...
@overload
async def async_wrap_sync_func(func: Callable[_P, Awaitable[_R]], *args: _P.args, **kwargs: _P.kwargs)->_R:...
@overload
async def async_wrap_sync_func(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs)->_R:...

async def async_wrap_sync_func(func, *args, **kwargs):
    '''
    Wrap a sync function to async function,
    i.e. passing it to run in threadpool, and wait for the result.
    If async function is passed in, it will be awaited directly.
    '''
    if is_async_callable(func):
        return await func(*args, **kwargs)
    
    from .helper_funcs import get_threadpool, run_func_with_context
    pool = get_threadpool()
    future = pool.submit(
        run_func_with_context,
        func,
        contextvars.copy_context(),  # type: ignore 
        *args, 
        **kwargs
    )
    while not future.done():
        await asyncio.sleep(0.1)
    if future.cancelled():
        _logger.debug(f'(async_wrap_sync_func) Future of func `{func}` was cancelled')
        raise RuntimeError('Future was cancelled')
    if (e:=future.exception()):
        _logger.warning(f'(async_wrap_sync_func) Future of func `{func}` raised exception: {e}')
        raise e    # type: ignore
    return future.result()  # type: ignore

async def async_wrap_sync_func_with_timeout(func: SyncOrAsyncFunc, args: tuple, kwargs: dict[str, Any], timeout: int|float|None = None):
    '''
    (with timeout version) Wrap a sync function to async function,
    i.e. passing it to run in threadpool, and wait for the result.
    If async function is passed in, it will be awaited directly.
    '''
    if is_async_callable(func):
        return await func(*args, **kwargs)
    if not timeout or timeout<=0:
        return await async_wrap_sync_func(func, *args, **kwargs)
    
    from .helper_funcs import get_threadpool, run_func_with_context
    pool = get_threadpool()
    future = pool.submit(
        run_func_with_context,
        func,
        contextvars.copy_context(),  # type: ignore 
        *args, 
        **kwargs
    )
    time_count = 0
    
    while not future.done():
        await asyncio.sleep(0.1)
        time_count += 0.1
        if time_count >= timeout:
            future.cancel()
            _logger.debug(f'(async_wrap_sync_func) Future of func `{func}` was cancelled due to timeout')
            raise TimeoutError(f'Future {func} was cancelled due to timeout')
        
    if future.cancelled():
        _logger.debug(f'(async_wrap_sync_func) Future of func `{func}` was cancelled')
        raise RuntimeError('Future was cancelled')
    if (e:=future.exception()):
        _logger.warning(f'(async_wrap_sync_func) Future of func `{func}` raised exception: {e}')
        raise e    # type: ignore
    return future.result()  # type: ignore
    
SyncOrAsyncGenerator = TypeAliasType("SyncOrAsyncGenerator", AsyncGenerator[_R, None]|Generator[_R, None, None], type_params=(_R,))
'''A type that can be either an async generator or a sync generator.'''
_GenerableT = TypeAliasType('_GenerableT', SyncOrAsyncGenerator[_R]|Iterable[_R]|AsyncIterable[_R], type_params=(_R,))

async def get_async_generator(
    gen: _GenerableT[_R]|Coroutine[_GenerableT[_R], None, None]
)->AsyncGenerator[_R, None]:
    '''
    Get async generator from sync/async generator/iterable.
    If a coroutine is passed in, it will be awaited. But the return value
    must be a generator.
    '''
    if isinstance(gen, Coroutine):
        gen = await gen # type: ignore
    if not isinstance(gen, (Iterable, AsyncIterable)):  # this also includes generator
        raise TypeError(f'Expected generator or iterable, but got {type(gen)}')
    if isinstance(gen, AsyncIterable):
        async for item in gen:
            yield item
    else:
        for item in gen:
            yield item

async def async_enumerate(gen: _GenerableT[_R], start: int = 0) -> AsyncGenerator[tuple[int, _R], None]:
    '''
    Async version of `enumerate`.
    NOTE: if you are passing a sync generator/iterable, it will also be converted to async generator.
    '''
    count = start
    gen = get_async_generator(gen)  # type: ignore
    async for y in gen:
        yield count, y
        count += 1


__all__ = [
    'run_async_in_sync', 
    'run_async_funcs', 
    'run_any_func', 
    'run_any_func_with_timeout', 
    'async_run_any_func', 
    'async_wrap_sync_func',
    'async_wrap_sync_func_with_timeout',
    'SyncOrAsyncGenerator',
    'get_async_generator',
    'async_enumerate',
    'async_run',
    'wait_coroutine',
]



if __name__ == '__main__':
    def test_run_any_func():
        async def f():
            for i in range(3):
                await asyncio.sleep(0.25)
                print('f', i)
            return 'f exit'
        
        async def g():
            print('g')
            await asyncio.sleep(0.5)
            r = run_any_func(f)
            print('f return:', r)
            return 'g exit'
        
        def sync_f():
            r = run_any_func(g)
            print('g return:', r)
        
        sync_f()
        
        class CustomError(Exception):...
    
        async def raise_err_test():
            await asyncio.sleep(0.1)
            raise CustomError('Test error')

        try:
            run_any_func(raise_err_test)    # type: ignore
        except CustomError as e:
            print('Caught exception as expected:', e)
    
    test_run_any_func()