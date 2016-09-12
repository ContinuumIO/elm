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
def client_context(dask_client, dask_scheduler):
    global get_func
    if dask_client == 'DISTRIBUTED':
        assert Executor is not None, "You need to install distributed"
        client = Executor(dask_scheduler)
        get_func = client.get
    elif dask_client == 'THREAD_POOL':
        client = ThreadPool(DASK_THREADS)
        get_func = dask_threaded_get
    elif dask_client == 'SERIAL':
        client = None
        get_func = get_sync
    else:
        raise ValueError('Did not expect DASK_EXECUTOR to be {}'.format(dask_client))
    with da.set_options(pool=dask_client):
        if dask_client in ("THREAD_POOL",):
            yield client
        else:
            yield client


def wait_for_futures(futures, client=None):
    '''Abstraction of waiting for mapped results
    that works for any type of client or no client'''
    if hasattr(client, 'gather'): # distributed
        from distributed import progress
        progress(futures)
        results = client.gather(futures)
    else:
        results = list(futures)
    return results


def no_client_submit(func, *args, **kwargs):
    return func(*args, **kwargs)

__all__ = ['no_client_submit', 'client_context', 'wait_for_futures']