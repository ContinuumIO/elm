from collections import OrderedDict
import logging

import xarray as xr

from elm.readers.util import (_extract_valid_xy, Canvas,
                              geotransform_to_bounds,
                              dummy_canvas)

__all__ = ['ElmStore', ]

logger = logging.getLogger(__name__)

class ElmStore(xr.Dataset):
    _es_kwargs = {
                    'add_canvas': True,
                    'lost_axis': None
        }
    def __init__(self, *args, **kwargs):
        es_kwargs = {k: kwargs.pop(k, v)
                     for k,v in self._es_kwargs.items()}
        super(ElmStore, self).__init__(*args, **kwargs)
        self.attrs['_dummy_canvas'] = not es_kwargs['add_canvas']
        if es_kwargs['add_canvas']:
            self._add_band_order()
            self._add_es_meta()
        else:
            self._add_band_order()
            self._add_dummy_canvas(**es_kwargs)

    def _add_dummy_canvas(self, **es_kwargs):
        lost_axis = es_kwargs['lost_axis']
        for band in self.band_order:
            band_arr = getattr(self, band)
            shp = band_arr.values.shape
            if len(shp) < 2:
                if lost_axis == 0:
                    shp = (1, shp[0])
                elif lost_axis == 1:
                    shp = (shp[0], 1)
            band_arr.attrs['canvas'] = dummy_canvas(shp[1], shp[0],
                                                    band_arr.dims)

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

    def _add_band_order(self):
        '''Ensure es.band_order is consistent with es.data_vars'''
        new = []
        old = list(getattr(self, 'band_order', []))
        for band in self.data_vars:
            if band not in old and band != 'flat':
                new.append(band)
        self.attrs['band_order'] = old + new

    def _add_es_meta(self):
        band_order = getattr(self, 'band_order', sorted(self.data_vars))
        self.attrs['band_order'] = band_order
        if tuple(self.data_vars.keys()) != ('flat',):
            self._add_canvases()

    def _add_canvases(self):

        old_canvas = None
        shared = True
        band_arr = None
        for band in self.data_vars:
            if band == 'flat':
                continue
            band_arr = getattr(self, band)
            x, xname, y, yname = _extract_valid_xy(band_arr)
            z = getattr(band_arr, 'z', None)
            t = getattr(band_arr, 't', None)
            if x is not None:
                xsize = x.size
            else:
                xsize = None
            if y is not None:
                ysize = y.size
            else:
                ysize = None
            if z is not None:
                zsize = z.size
                zbounds = [np.min(z), np.max(z)]
            else:
                zsize = zbounds = None
            if t is not None:
                tsize = t.size
                tbounds = [np.min(t), np.max(t)]
            else:
                tsize = tbounds = None
            canvas = getattr(band_arr, 'canvas', getattr(self, 'canvas', None))
            if canvas is not None:
                geo_transform = canvas.geo_transform
            else:
                geo_transform = band_arr.attrs.get('geo_transform', None)
                if geo_transform is None:
                    geo_transform = getattr(band_arr, 'geo_transform', getattr(self, 'geo_transform'))
            band_arr.attrs['canvas'] = Canvas(**OrderedDict((
                ('geo_transform', geo_transform),
                ('ysize', ysize),
                ('xsize', xsize),
                ('zsize', zsize),
                ('tsize', tsize),
                ('dims', band_arr.dims),
                ('ravel_order', getattr(self, 'ravel_order', 'C')),
                ('zbounds', zbounds),
                ('tbounds', tbounds),
                ('bounds', geotransform_to_bounds(xsize, ysize, geo_transform)),
            )))
            if old_canvas is not None and old_canvas != band_arr.attrs['canvas']:
                shared = False
            old_canvas = band_arr.attrs['canvas']
        if shared and band_arr is not None:
            self.attrs['canvas'] = band_arr.canvas
            logger.debug('Bands share coordinates')

    def __str__(self):
        return "ElmStore:\n" + super().__str__()




