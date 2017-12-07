from __future__ import print_function, division

from collections import OrderedDict
import datetime
import getpass
import os

from elm.pipeline.steps import (linear_model,
                                decomposition,
                                gaussian_process)
from elm.pipeline.predict_many import predict_many
from pydap.cas.urs import setup_session
import numpy as np
import xarray as xr
from xarray_filters import MLDataset
from xarray_filters.pipeline import Step


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


def slice_nldas_forcing_a(date, X_time_steps=144, feature_layers=None, **kw):
    dates = []
    for hours_ago in range(X_time_steps):
        file_time = date - datetime.timedelta(hours=hours_ago)
        dates.append(file_time)
    paths = [get_file(date, name=FORA) for date in dates]
    fora = xr.open_mfdataset(paths[:1], concat_dim='time', engine='pynio')
    path = get_file(date, name=VIC)
    vic  = xr.open_dataset(path, engine='pynio')
    vic  = MLDataset(OrderedDict([(SOIL_MOISTURE, vic[SOIL_MOISTURE])]))
    dset = MLDataset(xr.merge((vic, fora)))
    return dset


def get_y(y_field, X, y=None, sample_weight=None, **kw):
    '''Get the VIC Y column out of a flattened Dataset
    of FORA and VIC DataArrays'''
    print('X', X.data_vars.keys(), X.features.layer)
    y = X.features.sel(layer=y_field)
    features = X.features.sel(layer=[x for x in X.features.layer.values
                                     if x != y_field])
    X2 = MLDataset(OrderedDict([('features', features)]),
                   attrs=X.attrs)
    return X2, y


class GetY(Step):
    column = SOIL_MOISTURE
    def transform(self, X, y=None, **kw):
        X, y = X
        return get_y(self.column, X, **self.get_params())

