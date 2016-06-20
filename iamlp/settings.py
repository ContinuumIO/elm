import os

from dask import delayed as delayed
from toolz import curry

if os.environ.get('SERIAL_EVAL'):
    @curry
    def delayed(func, **k):
        def new_func(*args, **kwargs):
            return func(*args, **kwargs)
        return new_func
    SERIAL_EVAL = True
else:
    SERIAL_EVAL = False
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", '.')
