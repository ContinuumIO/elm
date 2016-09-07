'''

'''

import logging

import xarray as xr

from elm.readers.util import add_es_meta

__all__ = ['ElmStore', ]

logger = logging.getLogger(__name__)

class ElmStore(xr.Dataset):

    def __init__(self, *args, **kwargs):
        super(ElmStore, self).__init__(*args, **kwargs)

    def select_canvas(self, canvas):
        from elm.readers.reshape import select_canvas_elm_store
        return select_canvas_elm_store(self, canvas)

    def drop_na_rows(self):
        from elm.readers.reshape import drop_na_rows
        assert self.is_flat()
        return drop_na_rows(self)

    def flatten(self, ravel_order='C'):
        from elm.readers.reshape import flatten
        flat = flatten(self, ravel_order=ravel_order)
        assert flat.is_flat()
        return flat

    def filled_flattened(self):
        from elm.readers.reshape import filled_flattened
        assert self.is_flat()
        return filled_flattened(self)

    def inverse_flatten(self, **attrs):
        from elm.readers.reshape import inverse_flatten, check_is_flat
        assert self.is_flat()
        inverse = inverse_flatten(self, **attrs)
        assert not check_is_flat(inverse, raise_err=False)
        return inverse

    def get_shared_canvas(self):
        canvas = getattr(self, 'canvas', None)
        if canvas is not None:
            return canvas
        old_canvas = None
        shared = True
        for band in self.data_vars:
            canvas = getattr(self, band).canvas
            if canvas == old_canvas or old_canvas is None:
                pass
            else:
                shared = False
                break
            old_canvas = canvas
        return (canvas if shared else None)

    def is_flat(self):
        from elm.readers.reshape import check_is_flat
        return check_is_flat(self, raise_err=False)

    def agg(self, **kwargs):
        from elm.readers.reshape import aggregate_simple
        return aggregate_simple(self, **kwargs)

    def _transpose(self, new_dims):
        from elm.readers.reshape import transpose
        return transpose(self, new_dims)

    def add_band_order(self):
        '''Ensure es.band_order is consistent with es.data_vars'''
        new = []
        old = getattr(self, 'band_order', None) or []
        for band in self.data_vars:
            if band not in old:
                new.append(band)
        self.attrs['band_order'] = old + new

    def __str__(self):
        return "ElmStore:\n" + super().__str__()




