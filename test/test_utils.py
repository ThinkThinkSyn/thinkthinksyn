import os
import sys
import time
import dotenv
import asyncio
import logging
import inspect
import contextlib
import colorlog

from functools import partial, cache, wraps
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Callable, Any

formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s]%(reset)s %(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'white,dim',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
        'SUCCESS': 'green',
    }
)
logging.addLevelName(100, "SUCCESS")
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

class _Logger(logging.Logger):
    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(100):
            self._log(100, msg, args, **kwargs)
logging.setLoggerClass(_Logger)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

env_path = os.path.join(project_dir, '.env')
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path)

logger: _Logger = logging.getLogger() # type: ignore
logger.handlers = [_log_handler]
setattr(logger, 'success', partial(_Logger.success, logger))

@contextlib.contextmanager
def _timer(label: str):
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{label} took {end - start:.4f} seconds")

_T = TypeVar('_T', bound=Callable[[], Any])

def test_func(func: _T)-> _T:
    func_name = func.__name__   # type: ignore
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs): # type: ignore
            with _timer(func_name):
                try:
                    result = await func(*args, **kwargs)
                    logger.success(f"`{func_name}` passed. Result: \n{result}\n")
                except Exception as e:
                    logger.error(f"`{func_name}` failed with exception: {e}")
    else:
        @wraps(func)    # type: ignore
        def wrapper(*args, **kwargs):
            with _timer(func_name):
                try:
                    result = func(*args, **kwargs)  # type: ignore
                    logger.success(f"`{func_name}` passed. Result: \n{result}\n")
                except Exception as e:
                    logger.error(f"`{func_name}` failed with exception: {e}")
    return wrapper  # type: ignore

_testing_funcs: dict[str, list[Callable[[], Any]]] = {}

def register_testing(module: str)->Callable[[_T], _T]:
    if module not in _testing_funcs:
        _testing_funcs[module] = []
    def decorator(func: _T)-> _T:
        _testing_funcs[module].append(func)
        return func
    return decorator

@cache
def _thread_pool_executor()->ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

def run_testing(module: str):
    if module not in _testing_funcs:
        logger.info(f"No testing functions registered for module '{module}'")
        return
    sync_funcs, async_funcs = [], []
    for func in _testing_funcs[module]:
        if inspect.iscoroutinefunction(func):
            async_funcs.append(test_func(func))  # type: ignore
        else:
            sync_funcs.append(test_func(func))  # type: ignore
    total_count = len(sync_funcs) + len(async_funcs)
    if async_funcs:
        def run_async_tests(async_funcs):
            async def run_async_tests():
                await asyncio.gather(*(func() for func in async_funcs))
            asyncio.run(run_async_tests())
        sync_funcs.append(partial(run_async_tests, async_funcs))
    
    with _timer(f"{total_count} tests for module '{module}'"):
        futures = []
        for func in sync_funcs:
            futures.append(_thread_pool_executor().submit(func))
        for future in futures:
            future.result()


from thinkthinksyn import ThinkThinkSyn
tts = ThinkThinkSyn(apikey=os.getenv('TTS_APIKEY', ''))

__all__ = [
    'logger',
    'test_func',
    'register_testing',
    'tts',
]