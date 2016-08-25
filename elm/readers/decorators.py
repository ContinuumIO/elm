from functools import wraps

from elm.readers.reshape import drop_na_rows, flatten, inverse_flatten

__all__ = ['data_arrays_as_columns',
           'columns_as_data_arrays',]
def data_arrays_as_columns(func):
    '''Decorator to require that an ElmStore is flattened
    to 2-d (bands as columns)'''
    @wraps(func)
    def new_func(es, *args, **kwargs):
        flat = flatten(es)
        na_dropped = drop_na_rows(flat)
        return func(na_dropped, *args, **kwargs)
    return new_func


def columns_as_data_arrays(func):
    @wraps(func)
    def new_func(flat_filled, *args, **kwargs):
        return inverse_flatten(flat_filled)
    return new_func
