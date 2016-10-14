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

from elm.config.env import parse_env_vars

SERIAL_EVAL = None # reset by elm.config.load_config.ConfigParser

get_func = None

def _find_get_func_for_client(client):
    if client is None:
        return get_sync
    elif Executor and isinstance(client, Executor):
        return client.get
    elif isinstance(client, ThreadPool):
        return dask_threaded_get
    else:
        raise ValueError('client argument not a thread pool dask scheduler or None')

@contextlib.contextmanager
def client_context(dask_client=None, dask_scheduler=None):
    global get_func
    env = parse_env_vars()
    dask_client = dask_client or env.get('DASK_CLIENT', 'SERIAL')
    dask_scheduler = dask_scheduler or env.get('DASK_SCHEDULER')
    if dask_client == 'DISTRIBUTED':
        if Executor is None:
            raise ValueError('distributed is not installed - "conda install distributed"')
        client = Executor(dask_scheduler)
    elif dask_client == 'THREAD_POOL':
        client = ThreadPool(DASK_THREADS)
    elif dask_client == 'SERIAL':
        client = None
    else:
        raise ValueError('Did not expect DASK_EXECUTOR to be {}'.format(dask_client))
    get_func = _find_get_func_for_client(client)
    with da.set_options(pool=dask_client):
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