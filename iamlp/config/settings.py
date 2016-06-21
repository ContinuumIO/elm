import contextlib
import dask.array as da
import os

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

from dask import delayed as delayed
from toolz import curry

EXE_TYPES = ('SERIAL', 'THREAD_POOL',
             'PROCESS_POOL', 'DISTRIBUTED')

DOWNLOAD_DIR = os.environ.get('DOWNLOAD_DIR', '.')
DASK_EXECUTOR = os.environ.get('DASK_EXECUTOR', 'DISTRIBUTED')
DASK_SCHEDULER = os.environ.get('DASK_SCHEDULER', '127.0.0.1:8786')
DASK_PROCESSES = int(os.environ.get('DASK_PROCESSES', os.cpu_count()))
DASK_THREADS = int(os.environ.get('DASK_THREADS', os.cpu_count()))
SERIAL_EVAL = DASK_EXECUTOR == 'SERIAL'
@contextlib.contextmanager
def executor_context():
    try:
        if DASK_EXECUTOR == 'DISTRIBUTED':
            from distributed import Executor
            executor = Executor(DASK_SCHEDULER)
            get_func = executor.get
        elif DASK_EXECUTOR == 'PROCESS_POOL':
            executor = Pool(DASK_PROCESSES)
        elif DASK_EXECUTOR == 'THREAD_POOL':
            executor = ThreadPool(DASK_THREADS)
        else:
            assert DASK_EXECUTOR == 'SERIAL'
            executor = None
            get_func = None
        if DASK_EXECUTOR in ("PROCESS_POOL", "THREAD_POOL"):
            with da.set_options(pool=executor):
                get_func = executor.apply_async
                # TODO this isn't working
                yield executor, get_func
        else:
            yield executor, get_func
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
if SERIAL_EVAL:
    @curry
    def delayed(func, **k):
        def new_func(*args, **kwargs):
            return func(*args, **kwargs)
        return new_func
print('DASK_EXECUTOR', DASK_EXECUTOR)
