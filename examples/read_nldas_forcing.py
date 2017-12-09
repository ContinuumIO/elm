from __future__ import print_function, division

from collections import OrderedDict
import datetime
import getpass
import os

from elm.pipeline.predict_many import predict_many
from elm.pipeline.steps import (linear_model,
                                decomposition,
                                gaussian_process)
from pydap.cas.urs import setup_session
from xarray_filters import MLDataset
from xarray_filters.pipe_utils import for_each_array
from xarray_filters.pipeline import Step
import numpy as np
import pandas as pd
import xarray as xr


META_URL = 'https://cmr.earthdata.nasa.gov/search/granules.json?echo_collection_id=C1233767589-GES_DISC&sort_key%5B%5D=-start_date&page_size=20'

VIC, FORA = ('NLDAS_VIC0125_H', 'NLDAS_FORA0125_H',)

SOIL_MOISTURE = 'SOIL_M_110_DBLY'
FEATURE_LAYERS = [  # FORA DataArray's that may be differenced
    'A_PCP_110_SFC_acc1h',
    'PEVAP_110_SFC_acc1h',
    'TMP_110_HTGL',
    'DSWRF_110_SFC',
    'PRES_110_SFC',
    'DLWRF_110_SFC',
    'V_GRD_110_HTGL',
    'SPF_H_110_HTGL',
    'U_GRD_110_HTGL',
    'CAPE_110_SPDY',
]

FEATURE_LAYERS_CHOICES = [
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'DSWRF_110_SFC',
        'PRES_110_SFC',
        'DLWRF_110_SFC',
        'V_GRD_110_HTGL',
        'SPF_H_110_HTGL',
        'U_GRD_110_HTGL',
        'CAPE_110_SPDY',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'PRES_110_SFC',
        'DLWRF_110_SFC',
        'WIND_MAGNITUDE',
        'SPF_H_110_HTGL',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'WIND_MAGNITUDE',
        'SPF_H_110_HTGL',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'PRES_110_SFC',
        'WIND_MAGNITUDE',
        'SPF_H_110_HTGL',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'PRES_110_SFC',
        'WIND_MAGNITUDE',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'PRES_110_SFC',
        'SPF_H_110_HTGL',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'SPF_H_110_HTGL',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
        'TMP_110_HTGL',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'TMP_110_HTGL',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'PEVAP_110_SFC_acc1h',
    ],
    [
        'A_PCP_110_SFC_acc1h',
        'TMP_110_HTGL',
        'SPF_H_110_HTGL',
    ],

]

WIND_YX = ['U_GRD_110_HTGL', 'V_GRD_110_HTGL']

VIC, FORA = ('NLDAS_VIC0125_H', 'NLDAS_FORA0125_H',)

WATER_MASK = -9999

BASE_URL = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/{}/{:04d}/{:03d}/{}'
BASE_URL = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/{}/{:04d}/{:03d}/{}'

def get_session():
    username = os.environ.get('NLDAS_USERNAME') or raw_input('NLDAS Username: ')
    password = os.environ.get('NLDAS_PASSWORD') or getpass.getpass('Password: ')
    session = setup_session(username, password)
    return session

SESSION = get_session()


def make_url(year, month, day, hour, name, nldas_ver='002'):
    '''For given date components, data set identifier,
    and NLDAS version, return URL and relative path for a file

    Returns:
        url: URL on hydro1.gesdisc.eosdis.nasa.gov
        rel: Relative path named like URL pattern
    '''
    start = datetime.datetime(year, 1, 1)
    actual = datetime.datetime(year, month, day)
    julian = int(((actual - start).total_seconds() / 86400) + 1)
    vic_ver = '{}.{}'.format(name, nldas_ver)
    fname_pat = '{}.A{:04d}{:02d}{:02d}.{:04d}.{}.grb'.format(name, year, month, day, hour * 100, nldas_ver)
    url = BASE_URL.format(vic_ver, year, julian, fname_pat)
    rel = os.path.join('{:04d}'.format(year),
                       '{:03d}'.format(julian),
                       fname_pat)
    return url, os.path.abspath(rel)


def get_file(date, name, **kw):
    '''Pass date components and name arguments to make_url and
    download the file if needed.  Return the relative path
    in either case

    Parameters:
        See make_url function above: Arguments are passed to that function

    Returns:
        rel:  Relative path
    '''
    year, month, day, hour = date.year, date.month, date.day, date.hour
    url, rel = make_url(year, month, day, hour, name, **kw)
    path, basename = os.path.split(rel)
    if not os.path.exists(rel):
        if not os.path.exists(path):
            os.makedirs(path)
        print('Downloading', url, 'to', rel)
        r = SESSION.get(url)
        with open(rel, 'wb') as f:
            f.write(r.content)
    return rel


def nan_mask_water(arr, mask_value=WATER_MASK):
    # TODO is this function needed?
    arr.values[arr.values == mask_value] = np.NaN
    return arr


def wind_magnitude(fora):
    v, u = WIND_YX
    v, u = fora[v], fora[u]
    return (v ** 2 + u ** 2) ** (1 / 2.)


def _preprocess_vic(dset, field=SOIL_MOISTURE):
    arr = nan_mask_water(dset.data_vars[field])
    arr.attrs['source'] = 'VIC'
    return MLDataset(OrderedDict([(field, arr)]))


@for_each_array
def _preprocess_fora(arr):
    attrs = arr.attrs.copy()
    t = attrs.pop('initial_time')
    time = pd.Timestamp(t.replace(')','').replace('(', ''))
    attrs['time'] = time
    attrs['source'] = 'FORA'
    arr = arr.copy()
    arr.attrs.update(attrs)
    return arr


def slice_nldas_forcing_a(date, hours_back=4, **kw):
    dates = [date]
    for hours_back in range(1, hours_back):
        file_time = date - datetime.timedelta(hours=hours_back)
        dates.append(file_time)
    paths = [get_file(date, name=FORA) for date in dates]
    fora = xr.open_mfdataset(paths, concat_dim='time',
                             engine='pynio', chunks={},
                             preprocess=_preprocess_fora)
    fora = OrderedDict(fora.data_vars)
    fora['WIND_MAGNITUDE'] = wind_magnitude(fora)
    for layer, arr in fora.items():
        nan_mask_water(arr)
    paths = [get_file(date, name=VIC) for date in dates]
    vic  = xr.open_mfdataset(paths, engine='pynio',
                             concat_dim='time', chunks={},
                             preprocess=_preprocess_vic)
    nan_mask_water(vic.data_vars[SOIL_MOISTURE])
    fora[SOIL_MOISTURE] = vic.data_vars[SOIL_MOISTURE]
    dset = MLDataset(fora)
    dset.load()
    return dset


class GetY(Step):
    column = SOIL_MOISTURE
    def transform(self, X, y=None, **kw):
        feat = X.to_features().dropna(dim='space', how='any')
        idx = np.where(feat.features.layer.values == self.column)
        idx2 = np.where(feat.features.layer.values != self.column)[0]
        X = MLDataset(OrderedDict([('features', feat.features.isel(layer=idx2))]))
        y = feat.features.values[:, idx]
        return X, y

    fit_transform = transform
