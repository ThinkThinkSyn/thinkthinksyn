import os
import time

from io import TextIOWrapper
from typing import no_type_check
from abc import ABC, abstractmethod

if os.name != 'nt':
    import fcntl
    _TEMP_DIR = os.getenv('TMPDIR', '/tmp')
else:
    _TEMP_DIR = os.getenv('TEMP', '/tmp')
_DEFAULT_LOCK_DIR = ".cross_process_file_lock"

_BACKOFF = 0.001
_MAX_BACKOFF = 0.01

class _FileCrossProcessLockBase(ABC):
    '''Base class for cross-process file locks with shared functionality.'''

    __slots__ = ('lock_id', 'lock_dir', 'lock_file', 'fd')

    lock_id: str
    lock_dir: str
    lock_file: str
    fd: TextIOWrapper|None
    
    def __init__(
        self, 
        name: str, 
        lock_dir: str|None=None,
        suffix: str = ".lock"
    ):
        '''
        Args:
            name (str): Unique identifier for the lock.
            lock_dir (str|None): Directory to store lock files. 
                Defaults to a subdirectory in the system's temp directory.
            suffix (str): File suffix for lock files.
        '''
        self.lock_id = str(name)
        if not lock_dir:
            self.lock_dir = os.path.join(_TEMP_DIR, _DEFAULT_LOCK_DIR)
        else:
            # check if starts with `{temp_dir}`
            lock_dir = os.path.abspath(lock_dir)
            if not os.path.isabs(lock_dir):
                raise ValueError(f"lock_dir must be an absolute path, got {lock_dir}")
            if not lock_dir.startswith(_TEMP_DIR):
                lock_dir = os.path.join(_TEMP_DIR, lock_dir.lstrip('/\\'))
            self.lock_dir = lock_dir
        os.makedirs(self.lock_dir, exist_ok=True)
        self.lock_file = os.path.join(self.lock_dir, f"{self.lock_id}{suffix}")
        self.fd = None

    def wait(self, timeout: int|float|None=None)->bool:
        if os.name == 'nt':
            t = _BACKOFF
            start_time = time.time()
            while True:
                if not os.path.exists(self.lock_file):
                    return True
                if timeout and (time.time() - start_time) >= timeout:
                    return False
                time.sleep(t)
                t = min(t * 2, _MAX_BACKOFF)
        else:
            # Unix-like 系统使用 fcntl
            fd = None
            
            @no_type_check
            def check_unlocked():
                nonlocal fd
                if not os.path.exists(self.lock_file):
                    return True
                if fd is None:
                    fd = open(self.lock_file, 'r')
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    fd.close()
                    fd = None
                    return True
                except (IOError, BlockingIOError):
                    return False

            t = _BACKOFF
            start_time = time.time()
            while True:
                if check_unlocked():
                    if fd is not None:
                        fd.close()
                    return True
                if timeout and (time.time() - start_time) >= timeout:
                    if fd is not None:
                        fd.close()
                    return False
                time.sleep(t)
                t = min(t * 2, _MAX_BACKOFF)
    
    @abstractmethod
    def acquire(
        self, 
        blocking: bool=True, 
        timeout: int|float|None=None
    )->bool: ...

    @abstractmethod
    def release(self) -> bool:...

    @property
    @abstractmethod
    def locked(self) -> bool:...

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FileCrossProcessLock(_FileCrossProcessLockBase):
    '''
    Cross-process lock using file locking mechanism with 
    platform-specific implementations for Unix-like systems & Windows.
    '''

    @property
    def locked(self):
        if os.name == 'nt':  # Windows
            return os.path.exists(self.lock_file)
        else:  # Unix-like
            if self.fd is not None:
                return True
            if not os.path.exists(self.lock_file):
                return False
            try:
                with open(self.lock_file, 'r') as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                return False
            except (IOError, BlockingIOError, OSError):
                return True
        return False
        
    def acquire(
        self, 
        blocking: bool=True, 
        timeout: int|float|None=None
    )->bool:
        '''
        Acquire the lock.
        Args:
            blocking (bool): If True, block until the lock is acquired. 
                             If False, return immediately if the lock cannot be acquired.
            timeout (int|float|None): Maximum time to wait for the lock in seconds. 
                                      If None, wait indefinitely (only if blocking is True).
        '''
        if self.fd is not None:
            return True
        if os.name == 'nt':  # Windows
            return self._acquire_windows(blocking, timeout)
        else:  # Unix-like
            return self._acquire_unix(blocking, timeout)
    
    @no_type_check
    def _acquire_unix(self, blocking: bool=True, timeout: float|int|None=None)->bool:
        try:
            self.fd = open(self.lock_file, 'w')
        except (IOError, OSError):
            return False
        
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        
        start_time = time.time()
        
        try:
            if blocking and timeout is not None:
                # For blocking with timeout, we need to implement our own timeout logic
                while True:
                    try:
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # Write PID after successfully acquiring the lock
                        self.fd.seek(0)
                        self.fd.write(str(os.getpid()))
                        self.fd.flush()
                        return True
                    except (IOError, BlockingIOError):
                        if time.time() - start_time >= timeout:
                            self.fd.close()
                            self.fd = None
                            return False
                        time.sleep(_BACKOFF)
            else:
                # Standard blocking or non-blocking
                fcntl.flock(self.fd, flags)
                # Write PID after successfully acquiring the lock
                self.fd.seek(0)
                self.fd.write(str(os.getpid()))
                self.fd.flush()
                return True
                
        except (IOError, BlockingIOError, OSError):
            self.fd.close()
            self.fd = None
            return False

    def _acquire_windows(self, blocking: bool=True, timeout: float|int|None=None)->bool:
        start_time = time.time()
        t = _BACKOFF
        while True:
            try:
                self.fd = open(self.lock_file, 'x')
                self.fd.write(str(os.getpid()))
                self.fd.flush()
                return True
            except (FileExistsError, PermissionError):
                # Check if we should timeout
                timed_out = timeout is not None and (time.time() - start_time >= timeout)
                
                if not blocking or timed_out:
                    return False
                    
                time.sleep(t)
                t = min(t * 2, _MAX_BACKOFF)
    
    def release(self) -> bool:
        if self.fd is None:
            return False
        
        success = True
        if os.name == 'nt':  # Windows
            self.fd.close()
            try:
                os.unlink(self.lock_file)
            except Exception:
                success = False
        else:  # Unix-like
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except Exception:
                success = False
            self.fd.close()
            try:
                os.unlink(self.lock_file)
            except Exception:
                success = False
        
        self.fd = None
        return success


__all__ = ['FileCrossProcessLock']


if __name__.endswith('main__'):
    def _worker_process(lock_name, worker_id, iterations, results_queue, test_params=None):
        """Worker process for multi-process testing"""
        import logging, time
        test_params = test_params or {}
        lock = FileCrossProcessLock(lock_name)
        success_count = 0
        
        for i in range(iterations):
            try:
                # Test different acquire parameters
                blocking = test_params.get('blocking', True)
                timeout = test_params.get('timeout', None)
                
                if lock.acquire(blocking=blocking, timeout=timeout):
                    logging.info(f"Worker {worker_id} acquired lock (iteration {i+1})")
                    
                    # Simulate some work
                    time.sleep(0.01)
                    
                    # Verify we still have the lock
                    if lock.locked:
                        success_count += 1
                    
                    lock.release()
                    logging.info(f"Worker {worker_id} released lock (iteration {i+1})")
                else:
                    logging.warning(f"Worker {worker_id} failed to acquire lock (iteration {i+1})")
                
                # Small delay between iterations
                time.sleep(0.001)
                
            except Exception as e:
                logging.error(f"Worker {worker_id} error in iteration {i+1}: {e}")
        
        results_queue.put((worker_id, success_count))

if __name__ == '__main__': # unit tests
    import time
    import multiprocessing
    import tempfile
    
    def test_lock_speed(count=1000):
        print('## Testing CrossProcessLock speed ##')
        lock = FileCrossProcessLock("my_lock_speed")
        
        start_time = time.time()
        for _ in range(count):
            lock.acquire()
            lock.release()
        end_time = time.time()
        print(f"Acquired and released lock {count} times in {end_time - start_time:.2f} seconds")

    def test_multiprocess_locking():
        """Test cross-process locking functionality"""
        print('\n## Testing Multi-Process Locking ##')
        
        lock_name = "multiprocess_test"
        num_processes = 4
        iterations_per_process = 5
        
        # Clean up any existing lock files
        lock = FileCrossProcessLock(lock_name)
        if lock.acquire(blocking=False):
            lock.release()
        
        # Create processes
        processes = []
        results_queue = multiprocessing.Queue()
        
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=_worker_process,
                args=(lock_name, i, iterations_per_process, results_queue)
            )
            processes.append(p)
        
        # Start all processes
        start_time = time.time()
        for p in processes:
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        end_time = time.time()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        total_successes = sum(success for _, success in results)
        expected_total = num_processes * iterations_per_process
        
        print(f"Total successful acquisitions: {total_successes}/{expected_total}")
        print(f"Test duration: {end_time - start_time:.2f} seconds")
        print(f"Results by worker: {dict(results)}")
        
        return total_successes == expected_total

    def test_blocking_behavior():
        """Test blocking and non-blocking behavior"""
        print('\n## Testing Blocking Behavior ##')
        
        lock_name = "blocking_test"
        lock = FileCrossProcessLock(lock_name)
        
        # Clean up
        lock.release()
        
        # Test 1: Non-blocking when lock is available
        success = lock.acquire(blocking=False)
        print(f"Non-blocking acquire when available: {'PASS' if success else 'FAIL'}")
        
        if success:
            # Test 2: Non-blocking when lock is held (should fail)
            lock2 = FileCrossProcessLock(lock_name)
            success2 = lock2.acquire(blocking=False)
            print(f"Non-blocking acquire when locked: {'PASS' if not success2 else 'FAIL'}")
            
            lock.release()
        
        # Test 3: Timeout behavior
        def timeout_test():
            lock3 = FileCrossProcessLock(lock_name)
            lock3.acquire()  # This will block
            
            start_time = time.time()
            lock4 = FileCrossProcessLock(lock_name)
            success = lock4.acquire(timeout=1.0)
            elapsed = time.time() - start_time
            
            lock3.release()
            
            print(f"Timeout test: {'PASS' if not success and 0.9 <= elapsed <= 1.1 else 'FAIL'} (elapsed: {elapsed:.2f}s)")
        
        timeout_test()

    def test_context_manager():
        """Test context manager functionality"""
        print('\n## Testing Context Manager ##')
        
        lock_name = "context_test"
        
        try:
            with FileCrossProcessLock(lock_name) as lock:
                print(f"Inside context manager, locked: {lock.locked}")
                # Verify another instance can't acquire
                lock2 = FileCrossProcessLock(lock_name)
                can_acquire = lock2.acquire(blocking=False)
                print(f"Another instance blocked: {'PASS' if not can_acquire else 'FAIL'}")
            
            # After exiting context, lock should be released
            lock3 = FileCrossProcessLock(lock_name)
            can_acquire = lock3.acquire(blocking=False)
            print(f"Lock released after context: {'PASS' if can_acquire else 'FAIL'}")
            if can_acquire:
                lock3.release()
                
        except Exception as e:
            print(f"Context manager test failed: {e}")

    def test_custom_lock_dir():
        """Test custom lock directory"""
        print('\n## Testing Custom Lock Directory ##')
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_lock_dir = os.path.join(temp_dir, "custom_locks")
            
            lock = FileCrossProcessLock("custom_dir_test", lock_dir=custom_lock_dir)
            lock.release()  # Ensure clean state
            success = lock.acquire()
            
            # Check if lock file was created in the right place
            lock_file_exists = os.path.exists(lock.lock_file)
            in_custom_dir = custom_lock_dir in lock.lock_file
            
            print(f"Custom directory test: {'PASS' if success and lock_file_exists and in_custom_dir else 'FAIL'}")
            print(f"Lock file location: {lock.lock_file}")
            
            if success:
                lock.release()

    def run_all_tests():
        """Run all tests"""
        print("=" * 60)
        print("Running FileCrossProcessLock Comprehensive Tests")
        print("=" * 60)
        
        # Basic functionality
        test_lock_speed(1000)
        
        # Multi-process tests
        success = test_multiprocess_locking()
        print(f"\nMulti-process test: {'PASS' if success else 'FAIL'}")
        
        # Behavioral tests
        test_blocking_behavior()
        test_context_manager()
        test_custom_lock_dir()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    run_all_tests()