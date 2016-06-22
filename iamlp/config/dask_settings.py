import contextlib
import dask.array as da
import os

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

from dask import delayed as delayed as dask_delayed
from toolz import curry

SERIAL_EVAL = None # reset by iamlp.config.load_config.ConfigParser


@contextlib.contextmanager
def executor_context(dask_executor, scheduler_url):
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
        with da.set_options(pool=executor):
            yield pool
    else:
        yield executor

@curry
def delayed(func, **dec_kwargs):
    def new_func(*args, **kwargs):
        if SERIAL_EVAL: # SERIAL_EVAL is set in the globals()
                        # of this module when iamlp.config.load_config.ConfigParser
                        # is called
            return dask_delayed(func, **dec_kwargs)(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return new_func

__all__ = ['delayed', 'executor_context', ]