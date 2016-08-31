import contextlib
import dask.array as da
import os

from concurrent.futures import as_completed
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

from dask import delayed as dask_delayed
from toolz import curry

SERIAL_EVAL = None # reset by elm.config.load_config.ConfigParser


@contextlib.contextmanager
def executor_context(dask_executor, dask_scheduler):
    if dask_executor == 'DISTRIBUTED':
        from distributed import Executor
        executor = Executor(dask_scheduler)
        get_func = executor.get
    elif dask_executor == 'PROCESS_POOL':
        pool = Pool(DASK_PROCESSES)
    elif dask_executor == 'THREAD_POOL':
        pool = ThreadPool(DASK_THREADS)
    elif dask_executor == 'SERIAL':
        executor = None
        get_func = None
    else:
        raise ValueError('Did not expect DASK_EXECUTOR to be {}'.format(dask_executor))
    if dask_executor in ("PROCESS_POOL", "THREAD_POOL"):
        with da.set_options(pool=dask_executor):
            yield pool
    else:
        yield executor


def wait_for_futures(futures, executor=None):
    '''Abstraction of waiting for mapped results
    that works for any type of executor or no executor'''
    if not executor:
        results = list(futures)
    elif hasattr(executor, 'gather'): # distributed
        from distributed import progress
        progress(futures)
        results = executor.gather(futures)
    else:
        results = []
        for fut in as_completed(futures):
            if fut.exception():
                raise ValueError(fut.exception())
            results.append(fut.result())
    return results

def no_executor_submit(func, *args, **kwargs):
    return func(*args, **kwargs)

__all__ = ['no_executor_submit', 'executor_context', 'wait_for_futures']