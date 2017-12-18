from __future__ import print_function, division

from collections import OrderedDict
import glob
import json
import os

from xarray_filters import MLDataset
import numpy as np
import pandas as pd
import xarray as xr
import yaml

SOIL_URL = 'https://ldas.gsfc.nasa.gov/nldas/NLDASsoils.php'

SOIL_META_FILE = os.path.abspath('soil_meta_data.yml')

with open(SOIL_META_FILE) as f:
    SOIL_META = yaml.safe_load(f.read())

SOIL_FILES = ('COS_RAWL',
              'HYD_RAWL',
              'HYD_CLAP',
              'HYD_COSB',
              'SOILTEXT',
              'STEX_TAB',
              'TXDM1',
              'PCNTS',)

BIN_FILE_META = {'NLDAS_Mosaic_soilparms.bin': '>f4',
                 'NLDAS_STATSGOpredomsoil.bin': '>i4',
                 'NLDAS_Noah_soilparms.bin': '>f4',
                }

SOIL_PHYS_FEATURES = (
 'HYD_RAWL_porosity',
 'COS_RAWL_hy_cond',
 'HYD_COSB_matric_potential',
 'SOILTEXT_fc',
 'COS_RAWL_wp',
 'HYD_RAWL_matric_potential',
 'COS_RAWL_porosity',
 'HYD_COSB_fc',
 'HYD_CLAP_b',
 'HYD_COSB_hy_cond',
 'HYD_COSB_porosity',
 'SOILTEXT_hy_cond',
 'HYD_RAWL_b',
 'SOILTEXT_wp',
 'COS_RAWL_matric_potential',
 'HYD_CLAP_porosity',
 'HYD_CLAP_matric_potential',
 'COS_RAWL_b',
 'SOILTEXT_matric_potential',
 'SOILTEXT_porosity',
 'HYD_COSB_b',
 'HYD_RAWL_hy_cond',
 'HYD_CLAP_hy_cond',
 'HYD_CLAP_wp',
 'COS_RAWL_fc',
 'HYD_RAWL_wp',
 'HYD_COSB_wp',
 'HYD_CLAP_fc',
 'HYD_RAWL_fc')

SOIL_FEAUTURES_CHOICES = {
    'HYD_RAWL': [f for f in SOIL_PHYS_FEATURES if 'HYD_RAWL' in f],
    'COS_RAWL':  [f for f in SOIL_PHYS_FEATURES if 'COS_RAWL' in f],
    'HYD_CLAP': [f for f in SOIL_PHYS_FEATURES if 'HYD_CLAP' in f]
}
SOIL_DIR = os.environ.get('SOIL_DATA', os.path.abspath('nldas_soil_inputs'))
if not os.path.exists(SOIL_DIR):
    os.mkdir(SOIL_DIR)
BIN_FILES = tuple(os.path.join(SOIL_DIR, 'bin', f)
                  for f in BIN_FILE_META)
parts = SOIL_DIR, 'asc', 'soils', '*{}*'
COS_HYD_FILES = {f: glob.glob(os.path.join(*parts).format(f))
                 for f in SOIL_FILES}

NO_DATA = -9.99
NO_DATA_BIN = -9999

def dataframe_to_rasters(df,
                         col_attrs=None,
                         drop_cols=None, keep_cols=None,
                         attrs=None,
                         new_dim=None,
                         new_dim_values=None):
    arrs = {}
    i, j, x, y = df.i, df.j, df.x, df.y
    i_pts, j_pts = np.max(i), np.max(j)
    coords = dict(y=np.unique(y), x=np.unique(x))
    coords[new_dim] = new_dim_values
    dims = ('y', 'x', 'horizon',)
    for col in df.columns:
        if col in ('i', 'j', 'x', 'y',):
            continue
        if not (drop_cols is None or col not in drop_cols):
            continue
        if not (keep_cols is None or col in keep_cols):
            continue
        arr = df[col].astype(np.float64)
        attrs = dict(meta=col_attrs[col])
        arr = arr.values.reshape(i_pts, j_pts, len(new_dim_values))
        arrs[col] = xr.DataArray(arr, coords=coords, dims=dims, attrs=attrs)
    return arrs


def read_ascii_grid(filenames, y, x, name, dsets=None):
    dsets = dsets or OrderedDict()
    template = np.empty((y.size, x.size, len(filenames)))
    coords = dict(y=y[::-1], x=x, horizon=list(range(1, 1 + len(filenames))))
    dims = ('y', 'x', 'horizon')
    attrs = dict(filenames=filenames)
    for idx, f in enumerate(filenames):
        template[:, :, idx] = np.loadtxt(f)
    dsets[name] = xr.DataArray(template, coords=coords,
                               dims=dims, attrs=attrs)
    return dsets


def read_one_ascii(f, names=None):
    df = pd.read_csv(f, sep='\s+', names=names, skiprows=0)
    return df


def _get_horizon_num(fname):
    ext = os.path.basename(fname).split('.')
    if ext[-1].isdigit():
        return int(ext[-1])
    return int(ext[0].split('_')[-1])


def read_binary_files(y, x, attrs=None, bin_files=None):
    raise NotImplementedError('See the TODO note below on why this function is not being used now')
    bin_files = bin_files or tuple(BIN_FILES)
    arrs = {}
    dims = 'y', 'x'
    attrs = attrs or {}
    coords = dict(y=y, x=x)
    for f in bin_files:
        basename = os.path.basename(f)
        name_token = basename.split('_')[1].split('predom')[0]
        dtype = BIN_FILE_META[basename]
        arr = np.fromfile(f, dtype=dtype).astype(np.float32)
        arr[np.isclose(arr, NO_DATA_BIN)] = np.NaN
        if basename in SOIL_META:
            names = SOIL_META[basename]
            max_texture = np.max(tuple(_[0] for _ in SOIL_META['TEXTURES']))
            arr[arr > max_texture] = np.NaN
            arr.resize(y.size, x.size, len(names))
            for idx, (name, meta) in enumerate(names):
                raster_name = '{}_{}'.format(name_token, name)
                att = dict(filenames=[f], field=[name], meta=meta)
                att.update(attrs.copy())
                arrs[raster_name] = xr.DataArray(arr[:, :, idx],
                                                 coords=coords,
                                                 dims=dims, attrs=att)
        else:
            arr.resize(y.size, x.size)
            att = dict(filenames=[f])
            att.update(attrs.copy())
            arrs[name_token] = xr.DataArray(arr, coords=coords,
                                            dims=dims, attrs=att)
    return MLDataset(arrs)


def read_ascii_groups(ascii_groups=None, to_raster=True):
    dsets = OrderedDict()
    to_concat_names = set()
    for name in (ascii_groups or sorted(COS_HYD_FILES)):
        fs = COS_HYD_FILES[name]
        if name.startswith(('COS_', 'HYD_',)):
            names = SOIL_META['COS_HYD']
        elif name.startswith(('TXDM', 'STEX',)):
            names = SOIL_META['SOIL_LAYERS']
            if name.startswith('TXDM'):
                read_ascii_grid(fs, *grid, name=name, dsets=dsets)
                continue
        col_headers = [x[0] for x in names]
        col_headers = [x[0] for x in names]
        exts = [_get_horizon_num(x) for x in fs]
        fs = sorted(fs)
        for idx, f in enumerate(fs, 1):
            df = read_one_ascii(f, col_headers)
            arrs = dataframe_to_rasters(df,
                                        col_attrs=dict(names),
                                        drop_cols=['i', 'j'],
                                        new_dim='horizon',
                                        new_dim_values=[idx])
            for column, v in arrs.items():
                column = '{}_{}'.format(name, column)
                dsets[(column, idx)] = v
                to_concat_names.add(column)
                if name.startswith('COS'):
                    grid = v.y, v.x
    for name in to_concat_names:
        ks = [k for k in sorted(dsets) if k[0] == name]
        arr = xr.concat(tuple(dsets[k] for k in ks), dim='horizon')
        if to_raster:
            arr = arr.mean(dim='horizon')
        dsets[name] = arr
        for k in ks:
            dsets.pop(k)
    for v in dsets.values():
        v.values[np.isclose(v.values, NO_DATA)] = np.NaN
    return MLDataset(dsets)


def read_nldas_soils(ascii_groups=None, bin_files=None, to_raster=True):
    ascii_groups = ascii_groups or sorted(COS_HYD_FILES)
    for a in (ascii_groups or []):
        if not a in COS_HYD_FILES:
            raise ValueErrror('ascii_groups contains {} not in {}'.format(a, set(COS_HYD_FILES)))
    dset_ascii = read_ascii_groups(ascii_groups, to_raster=to_raster)
    example = tuple(dset_ascii.data_vars.keys())[0]
    example = dset_ascii[example]
    y, x, dims = example.y, example.x, example.dims
    # TODO - Note read_binary_files is commented out
    #        I saw at least one data reading issue and
    #        and am not sure if it is fixed yet.
    #        The issue was a flipping north/south for one
    #        dataset, then for another I'm pretty sure
    #        it read the binary file with wrong assumption
    #        about single vs double and/or big vs little endian
    #   dset_bin = read_binary_files(y, x, bin_files=bin_files)
    #   return MLDataset(xr.merge((dset_bin, dset_ascii)))
    return dset_ascii


def download_data(session=None):
    if session is None:
        from read_nldas_forcing import get_session
    base_url, basename = os.path.split(SOIL_URL)
    fname = os.path.join(SOIL_DIR, basename.replace('.php', '.html'))
    if not os.path.exists(fname):
        response = get_session().get(SOIL_URL).content.decode().split()
        paths = [_ for _ in response if '.' in _
                 and 'href' in _.lower() and
                 (any(sf.lower() in _.lower() for sf in SOIL_FILES)
                  or '.bin' in _)]
        paths = [_.split('"')[1] for _ in paths]
        with open(fname, 'w') as f:
            f.write(json.dumps(paths))
    else:
        paths = json.load(open(fname))
    paths2 = []
    for path in paths:
        url = os.path.join(base_url, path)
        fname = os.path.join(SOIL_DIR, path.replace('../nldas', SOIL_DIR))
        paths2.append(fname)
        if not os.path.exists(fname):
            if not os.path.exists(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))
            content = get_session().get(url).content
            with open(fname, 'wb') as f:
                f.write(content)
    return paths2


_endswith = lambda x, end: x.endswith('_{}'.format(end))


def flatten_horizons(soils_dset, attrs=None):
    arrs = OrderedDict()
    attrs = attrs or soils_dset.attrs.copy()
    for k, v in soils_dset.data_vars.items():
        if 'horizon' in v.dims:
            arrs[k] = v.mean(dim='horizon')
        else:
            arrs[k] = v
    return MLDataset(arrs, attrs=attrs)


def soils_join_forcing(soils, X, subset=None):
    if subset:
        choices = SOIL_FEAUTURES_CHOICES[subset]
        soils = OrderedDict([(layer, arr)
                             for layer, arr in soils.data_vars.items()
                             if layer in choices])
        soils = MLDataset(soils)
    reidx = soils.reindex_like(X, method='nearest')
    return reidx.merge(X.rename(dict(lat_110='y', lon_110='x')),
                       compat='broadcast_equals')

if __name__ == '__main__':
    download_data()
    X = read_nldas_soils()

