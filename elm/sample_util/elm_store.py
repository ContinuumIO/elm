'''

-------------------------

``elm.readers.elm_store``
~~~~~~~~~~~~~~~~~~~~~~~~~

``elm.readers.ElmStore`` inherits from xarray.Dataset to provide named
"bands" or Datasets for satellite data.

When an ElmStore is created with a "geo_transform" key/value
in its attrs initialization argument, then an elm.readers.Canvas
object is constructed from the geo transform.  The Canvas attribute
is on each band, or xarray.DataArray, in the ElmStore because bands
may have different coordinates.

The Canvas object is used in elm for forcing different bands, DataArrays,
onto the same coordinate system, for example::

    from sklearn.cluster import KMeans
    from elm.readers import *
    from elm.pipeline import steps, Pipeline
    from elm.pipeline.tests.util import random_elm_store

    X = random_elm_store()
    selector = steps.SelectCanvas('band_1')
    flattener = steps.Flatten()
    pipe = Pipeline([selector, flattener, KMeans(n_clusters=2)])
    pipe.fit_ensemble(X, init_ensemble_size=3, ngen=1).predict_many(X)
'''

from collections import OrderedDict
import logging

import dask.array as da
import xarray as xr

from elm.sample_util.geo_transform import (_extract_valid_xy,
                                           Canvas,
                                           geotransform_to_bounds,
                                           dummy_canvas)


__all__ = ['ElmStore', ]

logger = logging.getLogger(__name__)



class ElmStore(xr.Dataset):
    '''ElmStore, an xarray.Dataset with a canvas attribute
    for rasters as bands and transformations of data for machine
    learning

    Parameters inhertited from xarray.Dataset:
        :data_vars: dict-like, optional
            A mapping from variable names to :py:class:`~xarray.DataArray`
            objects, :py:class:`~xarray.Variable` objects or tuples of the
            form ``(dims, data[, attrs])`` which can be used as arguments to
            create a new ``Variable``. Each dimension must have the same length
            in all variables in which it appears.
        :coords: dict-like, optional
            Another mapping in the same form as the `variables` argument,
            except the each item is saved on the dataset as a "coordinate".
            These variables have an associated meaning: they describe
            constant/fixed/independent quantities, unlike the
            varying/measured/dependent quantities that belong in `variables`.
            Coordinates values may be given by 1-dimensional arrays or scalars,
            in which case `dims` do not need to be supplied: 1D arrays will be
            assumed to give index values along the dimension with the same
            name.
        :attrs: dict-like, optional
            Global attributes to save on this dataset.
        :compat: {'broadcast_equals', 'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.

    Parameters unique to ElmStore are used internally in elm in
    :mod:`elm.readers.reshape`, including :func:`lost_axis` and :func:`add_canvas`.  See also
    :func:`elm.readers.reshape.inverse_flatten`

    ElmStore attrs:
        :canvas: elm.readers.Canvas object for elm.pipeline.steps.SelectCanvas
        :band_order: list of the band names in the order they will appear as columns
                    when steps.Flatten() is called to flatten raster DataArrays
                    to a single "flat" DataArray
    '''
    _es_kwargs = {
                    'add_canvas': True,
                    'lost_axis': None,
                }
    def __init__(self, *args, **kwargs):
        es_kwargs = {k: kwargs.pop(k, v)
                     for k, v in self._es_kwargs.items()}
        super(ElmStore, self).__init__(*args, **kwargs)
        self.attrs['_dummy_canvas'] = not es_kwargs['add_canvas']
        if es_kwargs['add_canvas']:
            self._add_band_order()
            self._add_es_meta()
        else:
            if not 'flat' in self.data_vars:
                self._add_band_order()
                self._add_dummy_canvas(**es_kwargs)

    def _add_dummy_canvas(self, **es_kwargs):
        '''Add a placeholder canvas if no geo_transform attr'''
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
                buf_xsize = x.size
            else:
                buf_xsize = None
            if y is not None:
                buf_ysize = y.size
            else:
                buf_ysize = None
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
                ('buf_ysize', buf_ysize),
                ('buf_xsize', buf_xsize),
                ('zsize', zsize),
                ('tsize', tsize),
                ('dims', band_arr.dims),
                ('ravel_order', getattr(self, 'ravel_order', 'C')),
                ('zbounds', zbounds),
                ('tbounds', tbounds),
                ('bounds', geotransform_to_bounds(buf_xsize, buf_ysize, geo_transform)),
            )))
            if old_canvas is not None and old_canvas != band_arr.attrs['canvas']:
                shared = False
            old_canvas = band_arr.attrs['canvas']
        if shared and band_arr is not None:
            self.attrs['canvas'] = band_arr.canvas
            logger.debug('Bands share coordinates')

    def plot_3d(self, bands, title='', scale=None,
                axis_labels=True, **imshow_kwargs):
        '''Plot a true or pseudo color image of 3 bands

        Parameters:
            :X: ElmStore or xarray.Dataset
            :bands: list of 3 band names that are in X
            :title: title for figure
            :scale: divide all values by this (e.g. 2** 16 for uint16)
            :axis_labels: True / False show axis_labels
            :\*\*imshow_kwargs: passed to imshow

        Returns:
            :(arr, fig): where arr is the 3-D numpy array and fig is the figure
        '''
        from elm.sample_util.plotting_helpers import plot_3d
        return plot_3d(self, bands, title, scale, axis_labels, **imshow_kwargs)

    def __str__(self):
        return "ElmStore:\n" + super().__str__().replace('xarray', 'elm')

    def __repr__(self):
        return "ElmStore:\n" + super().__repr__().replace('xarray', 'elm')


OK_X_DATA_TYPES = (ElmStore, xr.Dataset, da.Array)
def check_X_data_type(X):
    if not isinstance(X, OK_X_DATA_TYPES):
        raise ValueError('Expected the return value of fitting function '
                         'to be one of the following: '
                         '{} but found it was an {}'.format(OK_X_DATA_TYPES, type(X)))


__all__ = ['ElmStore', 'check_X_data_type',]