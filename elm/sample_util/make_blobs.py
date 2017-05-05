from collections import OrderedDict

from earthio import xy_canvas, ElmStore
import numpy as np
import xarray as xr
from sklearn.datasets import make_blobs

from elm.config import filter_kwargs_to_func


BANDS = ['band_{}'.format(idx + 1) for idx in range(40)]
GEO = [-2223901.039333, 926.6254330549998, 0.0, 8895604.157333, 0.0, -926.6254330549995]

def random_elm_store(bands=None, centers=None, std_devs=None, height=100, width=80, **kwargs):
    lenn = len(centers) if centers is not None else 3
    bands = bands or ['band_{}'.format(idx + 1) for idx in range(lenn)]
    if isinstance(bands, int):
        bands = ['band_{}'.format(idx + 1) for idx in range(bands)]
    if isinstance(bands[0], (list, tuple)):
        # it is actually band_specs
        bands = [_[-1] for _ in bands]
    default_centers = len(bands)
    if centers is None:
        centers = np.arange(100, 100 + len(bands) * default_centers).reshape((len(bands), default_centers))
    if std_devs is None:
        std_devs = np.ones((len(bands), default_centers))
    if len(centers) != len(bands) or len(bands) != len(std_devs):
        raise ValueError('Expected bands, centers, std_devs to have same length')
    if kwargs.get('attrs'):
        attrs = kwargs['attrs']
    else:
        attrs = {'width': width,
                 'height': height,
                 'geo_transform': GEO,
                 'canvas': xy_canvas(GEO, width, height, ('y', 'x'))}
    es_dict = OrderedDict()
    arr, y = make_blobs(n_samples=width * height, n_features=len(bands),
                        centers=centers, cluster_std=std_devs)
    for idx, band in enumerate(bands):
        es_dict[band] = xr.DataArray(arr[:, idx].reshape((height, width)),
                                     coords=[('y', np.arange(height)),
                                             ('x', np.arange(width))],
                                     dims=('y', 'x'),
                                     attrs=attrs)
    attrs['band_order'] = bands
    X = ElmStore(es_dict, attrs=attrs)
    if kwargs.get('return_y'):
        return X, y
    return X


def make_blobs_elm_store(**make_blobs_kwargs):
    '''sklearn.datasets.make_blobs - but return ElmStore
    Parameters:
        as_2d_or_3d:       int - 2 or 3 for num dimensions
        make_blobs_kwargs: kwargs for make_blobs, such as:
                           n_samples=100,
                           n_features=2,
                           centers=3,
                           cluster_std=1.0,
                           center_box=(-10.0, 10.0),
                           shuffle=True,
                           random_state=None'''
    kwargs = filter_kwargs_to_func(make_blobs, **make_blobs_kwargs)
    arr  = make_blobs(**kwargs)[0]
    band = ['band_{}'.format(idx) for idx in range(arr.shape[1])]
    es = ElmStore({'flat': xr.DataArray(arr,
                  coords=[('space', np.arange(arr.shape[0])),
                          ('band', band)],
                  dims=['space', 'band'],
                  attrs={'make_blobs': make_blobs_kwargs})})
    return es

