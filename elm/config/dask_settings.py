import contextlib
import dask.array as da
import os

from concurrent.futures import as_completed
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from dask.threaded import get as dask_threaded_get
from dask.async import get_sync

from dask import delayed as dask_delayed
from toolz import curry

try:
    from distributed import Executor
except ImportError:
    Executor = None

SERIAL_EVAL = None # reset by elm.config.load_config.ConfigParser

get_func = None

@contextlib.contextmanager
def executor_context(dask_executor, dask_scheduler):
    global get_func
    if dask_executor == 'DISTRIBUTED':
        assert Executor is not None, "You need to install distributed"
        executor = Executor(dask_scheduler)
        get_func = executor.get
    elif dask_executor == 'THREAD_POOL':
        executor = ThreadPool(DASK_THREADS)
        get_func = dask_threaded_get
    elif dask_executor == 'SERIAL':
        executor = None
        get_func = get_sync
    else:
        raise ValueError('Did not expect DASK_EXECUTOR to be {}'.format(dask_executor))
    with da.set_options(pool=dask_executor):
        if dask_executor in ("THREAD_POOL",):
            yield executor
        else:
            yield executor


def wait_for_futures(futures, executor=None):
    '''Abstraction of waiting for mapped results
    that works for any type of executor or no executor'''
    if hasattr(executor, 'gather'): # distributed
        from distributed import progress
        progress(futures)
        results = executor.gather(futures)
    else:
        results = list(futures)
    return results

def no_executor_submit(func, *args, **kwargs):
    return func(*args, **kwargs)

__all__ = ['no_executor_submit', 'executor_context', 'wait_for_futures']