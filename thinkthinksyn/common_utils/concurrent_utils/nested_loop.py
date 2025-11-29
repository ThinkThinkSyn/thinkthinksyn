"""
Patch asyncio to allow nested event loops.
Copy and modified from `nest_asyncio` package.
"""

import os
import sys
import asyncio
import threading
import logging
import asyncio.events as events

from typing import TYPE_CHECKING
from heapq import heappop
from asyncio import AbstractEventLoop
from contextlib import contextmanager, suppress

_logger = logging.getLogger(__name__)

class NestedLoopPolicy(asyncio.DefaultEventLoopPolicy):
    '''Special event loop policy to allow nested event loops.'''
    
    def get_or_create_event_loop(self) -> '_NestedLoop':
        if self._local._loop is None:   # type: ignore
            loop = self.new_event_loop()
            self.set_event_loop(loop)
        else:
            loop = self._local._loop    # type: ignore
        return loop
        
    def new_event_loop(self) -> "_NestedLoop":
        loop = super().new_event_loop()
        patch_loop(loop)
        return loop # type: ignore
    
    def get_event_loop(self):
        if self._local._loop is None:   # type: ignore
            loop = self.new_event_loop()
            self.set_event_loop(loop)
        return self._local._loop    # type: ignore

_nested_event_loop_policy = None

def get_nested_loop_policy() -> NestedLoopPolicy:
    '''Get the nested loop policy.'''
    global _nested_event_loop_policy
    if _nested_event_loop_policy is None:
        _nested_event_loop_policy = NestedLoopPolicy()
    return _nested_event_loop_policy

class _NestedLoop(AbstractEventLoop):
    '''
    The return value of `get_nested_loop`.
    This class is just for type hinting.
    '''
    
    _nest_patched: bool
    '''the flag to indicate whether the loop is patched'''
    
    if TYPE_CHECKING:
        @classmethod
        def __unpatch__(cls):
            '''Unpatch the loop to original state.'''
    
def patch_loop(loop: AbstractEventLoop):
    """Patch loop to make it reentrant."""
    if hasattr(loop, '_nest_patched') and loop._nest_patched:   # type: ignore
        return
    
    def run_forever(self):
        with manage_run(self), manage_asyncgens(self):  # type: ignore
            while True:
                self._run_once()
                if self._stopping:
                    break
        self._stopping = False

    def run_until_complete(self, future):
        with manage_run(self):
            f = asyncio.ensure_future(future, loop=self)
            if f is not future:
                f._log_destroy_pending = False
            while not f.done():
                self._run_once()
                if self._stopping:
                    break
            if not f.done():
                raise RuntimeError('Event loop stopped before Future completed.')
            return f.result()

    def _run_once(self):
        """
        Simplified re-implementation of asyncio's _run_once that
        runs handles as they become ready.
        """
        ready = self._ready
        scheduled = self._scheduled
        while scheduled and scheduled[0]._cancelled:
            heappop(scheduled)

        timeout = (
            0 if ready or self._stopping
            else min(max(
                scheduled[0]._when - self.time(), 0), 86400) if scheduled
            else None)
        event_list = self._selector.select(timeout)
        self._process_events(event_list)

        end_time = self.time() + self._clock_resolution
        while scheduled and scheduled[0]._when < end_time:
            handle = heappop(scheduled)
            ready.append(handle)

        for _ in range(len(ready)):
            if not ready:
                break
            handle = ready.popleft()
            if not handle._cancelled:
                # preempt the current task so that that checks in
                # Task.__step do not raise
                curr_task = curr_tasks.pop(self, None)

                try:
                    handle._run()
                finally:
                    # restore the current task
                    if curr_task is not None:
                        curr_tasks[self] = curr_task

        handle = None

    @contextmanager
    def manage_run(self):
        """Set up the loop for running."""
        self._check_closed()
        old_thread_id = self._thread_id
        old_running_loop = events._get_running_loop()
        try:
            self._thread_id = threading.get_ident()
            events._set_running_loop(self)
            self._num_runs_pending += 1
            if self._is_proactorloop:
                if self._self_reading_future is None:
                    self.call_soon(self._loop_self_reading)
            yield
        finally:
            self._thread_id = old_thread_id
            events._set_running_loop(old_running_loop)
            self._num_runs_pending -= 1
            if self._is_proactorloop:
                if (self._num_runs_pending == 0
                        and self._self_reading_future is not None):
                    ov = self._self_reading_future._ov
                    self._self_reading_future.cancel()
                    if ov is not None:
                        self._proactor._unregister(ov)
                    self._self_reading_future = None

    @contextmanager
    def manage_asyncgens(self):
        if not hasattr(sys, 'get_asyncgen_hooks'):
            # Python version is too old.
            return
        old_agen_hooks = sys.get_asyncgen_hooks()
        try:
            self._set_coroutine_origin_tracking(self._debug)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(
                    firstiter=self._asyncgen_firstiter_hook,
                    finalizer=self._asyncgen_finalizer_hook)
            yield
        finally:
            self._set_coroutine_origin_tracking(False)
            if self._asyncgens is not None:
                sys.set_asyncgen_hooks(*old_agen_hooks)

    def _check_running(self):
        """Do not throw exception if loop is already running."""
        pass

    if not isinstance(loop, asyncio.BaseEventLoop):
        raise ValueError('Can\'t patch loop of type %s' % type(loop))
    
    cls = loop.__class__
    cls.__original_run_forever__ = cls.run_forever  # type: ignore
    cls.run_forever = run_forever
    
    cls.__original_run_until_complete__ = cls.run_until_complete    # type: ignore
    cls.run_until_complete = run_until_complete
    
    cls.__original_run_once__ = cls._run_once   # type: ignore
    cls._run_once = _run_once   # type: ignore
    
    cls.__original_check_running__ = cls._check_running # type: ignore
    cls._check_running = _check_running     # type: ignore
    
    cls._num_runs_pending = 1 if loop.is_running() else 0   # type: ignore
    cls.__original_is_proactorloop__ = cls._is_proactorloop = (os.name == 'nt' and issubclass(cls, asyncio.ProactorEventLoop)) # type: ignore
    cls._is_proactorloop = (os.name == 'nt' and issubclass(cls, asyncio.ProactorEventLoop)) # type: ignore
    
    curr_tasks = asyncio.tasks._current_tasks if sys.version_info >= (3, 7, 0) else asyncio.Task._current_tasks     # type: ignore
    cls._nest_patched = True    # type: ignore
    
    @classmethod
    def unpatch(cls):
        if hasattr(cls, '__original_run_forever__'):
            cls.run_forever = cls.__original_run_forever__
            del cls.__original_run_forever__
        if hasattr(cls, '__original_run_until_complete__'):
            cls.run_until_complete = cls.__original_run_until_complete__
            del cls.__original_run_until_complete__
        if hasattr(cls, '__original_run_once__'):
            cls._run_once = cls.__original_run_once__
            del cls.__original_run_once__
        if hasattr(cls, '__original_check_running__'):
            cls._check_running = cls.__original_check_running__
            del cls.__original_check_running__
        if hasattr(cls, '__original_is_proactorloop__'):
            cls._is_proactorloop = cls.__original_is_proactorloop__
            del cls.__original_is_proactorloop__
        if hasattr(cls, '_nest_patched'):
            del cls._nest_patched
        del cls.__unpatch__
    cls.__unpatch__ = unpatch   # type: ignore
    
def get_nested_loop(
    patch_asyncio_module: bool=False,
    use_asyncio_origin_loop: bool=True,
)->_NestedLoop|None:
    '''
    Get the current event loop, and patch it if necessary.
    If no current event loop is running, create a new one.
    
    Args:
        - patch_asyncio_module: whether to patch the asyncio module. In that case,
                            `asyncio.run` and `asyncio.get_event_loop` will be
                            patched for supporting nested event loops.
        - use_asyncio_origin_loop: when creating a new loop, use the original
                            loop policy from asyncio module, to prevent
                            creating an un-patchable loop class(i.e. UVLoop).
    
    NOTE: if the current running event loop is a uvloop(which cannot be 
    patched), this function will return None.
    '''
    if patch_asyncio_module:
        patch_asyncio()
    
    if loop := asyncio._get_running_loop():
        from .async_helpers import _is_uvloop
        if _is_uvloop(loop):
            _logger.warning('Cannot patch asyncio when there is a running uvloop.')
            return None
    else:
        if use_asyncio_origin_loop:
            loop = _nested_event_loop_policy.new_event_loop()  # type: ignore
        else:
            loop = asyncio.new_event_loop()  # type: ignore
    
    patch_loop(loop)
    if not hasattr(loop, '_nest_patched') or not loop._nest_patched:    # type: ignore
        _logger.warning('Failed to patch the event loop.')
        return None
    return loop # type: ignore
        
def _patch_asyncio():
    """Patch asyncio module to use pure Python tasks and futures."""
    def run(main, *, debug=False):
        loop = asyncio.get_event_loop()
        loop.set_debug(debug)
        task = asyncio.ensure_future(main)
        try:
            return loop.run_until_complete(task)
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    loop.run_until_complete(task)

    def _get_event_loop(stacklevel=3):
        loop = events._get_running_loop()
        if loop is None:
            loop = events.get_event_loop_policy().get_event_loop()
        return loop

    # Use module level _current_tasks, all_tasks and patch run method.
    if hasattr(asyncio, '_nest_patched'):
        return
    
    if sys.version_info >= (3, 6, 0):
        asyncio.__original_Task__ = asyncio.Task    # type: ignore
        asyncio.__original_Future__ = asyncio.Future    # type: ignore
        asyncio.Task = asyncio.tasks._CTask = asyncio.tasks.Task = asyncio.tasks._PyTask    # type: ignore
        asyncio.Future = asyncio.futures._CFuture = asyncio.futures.Future = asyncio.futures._PyFuture  # type: ignore
    if sys.version_info >= (3, 9, 0):
        asyncio.__original_get_event_loop__ = asyncio.get_event_loop    # type: ignore
        events._get_event_loop = events.get_event_loop = asyncio.get_event_loop = _get_event_loop    # type: ignore
    
    asyncio.__original_run__ = asyncio.run        # type: ignore
    asyncio.run = run
    
    asyncio._nest_patched = True    # type: ignore

    def __unpatch__():
        if hasattr(asyncio, '__original_Task__'):
            asyncio.tasks._CTask = asyncio.tasks.Task = asyncio.Task = asyncio.__original_Task__    # type: ignore
            asyncio.tasks._CFuture = asyncio.futures.Future = asyncio.Future = asyncio.__original_Future__  # type: ignore
            del asyncio.__original_Task__     # type: ignore
            del asyncio.__original_Future__   # type: ignore
        if hasattr(asyncio, '__original_get_event_loop__'):
            asyncio.get_event_loop = events.get_event_loop = events._get_event_loop = asyncio.__original_get_event_loop__   # type: ignore
            del asyncio.__original_get_event_loop__     # type: ignore
        if hasattr(asyncio, '__original_run__'):
            asyncio.run = asyncio.__original_run__  # type: ignore
            del asyncio.__original_run__    # type: ignore
        if hasattr(asyncio, '_nest_patched'):
            del asyncio._nest_patched   # type: ignore
        del asyncio.__unpatch__ # type: ignore
    asyncio.__unpatch__ = __unpatch__   # type: ignore

def _patch_policy():
    """Patch the policy to always return a patched loop."""
    if hasattr(asyncio, '__unpatch_event_loop_policy__'):
        return
    
    def get_event_loop(self):
        if self._local._loop is None:
            loop = self.new_event_loop()
            patch_loop(loop)
            self.set_event_loop(loop)
        return self._local._loop

    policy = events.get_event_loop_policy()
    policy.__class__.__original_get_event_loop__ = policy.__class__.get_event_loop  # type: ignore
    policy.__class__.get_event_loop = get_event_loop    # type: ignore
    
    def __unpatch_event_loop_policy__():
        if hasattr(policy.__class__, '__original_get_event_loop__'):
            policy.__class__.get_event_loop = policy.__class__.__original_get_event_loop__  # type: ignore
            del policy.__class__.__original_get_event_loop__    # type: ignore
        del asyncio.__unpatch_event_loop_policy__   # type: ignore
        
    asyncio.__unpatch_event_loop_policy__ = __unpatch_event_loop_policy__   # type: ignore
    

def patch_asyncio()->bool:
    """
    Patch asyncio module to allow nested event loops.
    Return bool to indicate whether the patch is successful,
    i.e. when there is any running uvloop, this function
        will terminate without patching asyncio.
    """
    if hasattr(asyncio, '_nest_patched') and asyncio._nest_patched:  # type: ignore
        return getattr(asyncio, '_nest_patched')
    
    curr_loop = asyncio._get_running_loop()
    from .async_helpers import _is_uvloop
    if curr_loop and _is_uvloop(curr_loop):
        _logger.warning('Cannot patch asyncio when there is a running uvloop.')
        return False
    
    _patch_asyncio()
    _patch_policy()
    loop = asyncio.get_event_loop()
    patch_loop(loop)
    return True

def unpatch_asyncio():
    """Unpatch asyncio module to original state."""
    if not hasattr(asyncio, '_nest_patched'):
        return
    if hasattr(asyncio, '__unpatch__'):
        asyncio.__unpatch__()   # type: ignore
    if hasattr(asyncio, '__unpatch_event_loop_policy__'):
        asyncio.__unpatch_event_loop_policy__() # type: ignore
    if (curr_loop := asyncio._get_running_loop()):
        if hasattr(curr_loop, '__unpatch__'):
            curr_loop.__unpatch__() # type: ignore
    

__all__ = ['get_nested_loop', 'patch_asyncio', 'unpatch_asyncio', 'patch_loop', 'get_nested_loop_policy']
