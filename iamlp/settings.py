from dask import delayed
if os.environ.get('SERIAL_EVAL'):
    delayed = lambda func: func
    SERIAL_EVAL = True
else:
    SERIAL_EVAL = False
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", '.')
