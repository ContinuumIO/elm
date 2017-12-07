from __future__ import print_function, division
import dask
dask.set_options(get=dask.local.get_sync)

from collections import OrderedDict
import datetime
from functools import partial
from itertools import product
import os

import dill
from elm.pipeline import Pipeline
from elm.pipeline.steps import (linear_model,
                                decomposition,
                                gaussian_process,
                                preprocessing)
from elm.pipeline.predict_many import predict_many
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from elm.model_selection.sorting import pareto_front
from elm.model_selection import EaSearchCV
import numpy as np
from xarray_filters import MLDataset
from xarray_filters.pipeline import Generic, Step

from read_nldas_forcing import (slice_nldas_forcing_a,
                                GetY, FEATURE_LAYERS,
                                SOIL_MOISTURE)
from read_nldas_soils import download_data, read_nldas_soils
from nldas_soil_features import nldas_soil_features
from ts_raster_steps import differencing_integrating
from changing_structure import ChooseWithPreproc
import xarray as xr

NGEN = 3
NSTEPS = 1
WATER_MASK = -9999
DEFAULT_CV = 3
DEFAULT_MAX_STEPS = 12

START_DATE = datetime.datetime(2000, 1, 1, 1, 0, 0)

ONE_HR = datetime.timedelta(hours=1)
TIME_OPERATIONS = ('mean',
                   'std',
                   'sum',
                   ('diff', 'mean'),
                   ('diff', 'std'),
                   ('diff', 'sum'))
REDUCERS = [('mean', x) for x in TIME_OPERATIONS if x != 'mean']

np.random.seed(42)  # TODO remove

def log_trans_only_positive(self, X, y=None, **kw):
    Xnew = OrderedDict()
    for j in range(X.features.shape[1]):
        minn = X.features[:, j].min().values
        if minn <= 0:
            continue
        X.features.values[:, j] = np.log10(X.features.values[:, j])
    return X, y


class Flatten(Step):

    def transform(self, X, y=None, **kw):
        return X.to_features(), y

    fit_transform = transform


class Differencing(Step):
    hours_back = 144
    first_bin_width = 12
    last_bin_width = 1
    num_bins = 12
    weight_type = 'linear'
    reducers = 'mean'
    layers = None

    def transform(self, X, y=None, **kw):
        return differencing_integrating(X, **self.get_params())

    fit_transform = transform



SOIL_PHYS_CHEM = {}
class AddSoilPhysicalChemical(Step):
    add = True
    soils_dset = None
    to_raster = True
    avg_cos_hyd_params = False

    def transform(self, X, y=None, **kw):
        global SOIL_PHYS_CHEM
        params = self.get_params().copy()
        if not params.pop('add'):
            return X
        hsh = hash(repr(params))
        if hsh in SOIL_PHYS_CHEM:
            soils = SOIL_PHYS_CHEM[hsh]
        else:
            soils = nldas_soil_features(**params)
            soils = MLDataset(soils).to_features()
            if len(SOIL_PHYS_CHEM) < 3:
                SOIL_PHYS_CHEM[hsh] = soils
        return X[0].concat_ml_features()

    fit_transform = transform

SCALERS = [preprocessing.StandardScaler()] + [preprocessing.MinMaxScaler()] * 10
np.random.shuffle(SCALERS)
param_distributions = {
    'log__kw_args': [dict(trans_if=log_trans_only_positive),
                     dict(trans_if=None)],
    'scaler__feature_range': [(x, x * 2) for x in np.linspace(0, 1, 10)],
    'pca__n_components': [6, 7, 8, 10, 14, 18],
    'pca__estimator': [decomposition.PCA(),
                      decomposition.FastICA(),],
                      #decomposition.KernelPCA()],
    'pca__run': [True, True, False],
    'time__hours_back': [1],#list(np.linspace(1, DEFAULT_MAX_STEPS, 12).astype(np.int32)),
    'time__last_bin_width': [1,],
    'time__num_bins': [4,],
    'time__weight_type': ['uniform', 'log', 'log', 'linear', 'linear'],
    'time__weight_type': ['linear', 'log'],
    'time__reducers': REDUCERS,
    'soil_phys__add': [True, True, True, False],
}

model_selection = {
    'select_method': 'selNSGA2',
    'crossover_method': 'cxTwoPoint',
    'mutate_method': 'mutUniformInt',
    'init_pop': 'random',
    'indpb': 0.5,
    'mutpb': 0.9,
    'cxpb':  0.3,
    'eta':   20,
    'ngen':  2,
    'mu':    16,
    'k':     8, # TODO ensure that k is not ignored - make elm issue if it is
    'early_stop': None
}

def get_file_name(tag, date):
    date = date.isoformat().replace(':','_').replace('-','_')
    return '{}-{}.dill'.format(tag, date)


def dump(obj, tag, date):
    fname = get_file_name(tag, date)
    return getattr(obj, 'dump', getattr(obj, 'to_netcdf'))(fname)


class Sampler(Step):
    date = None
    def transform(self, dates, y=None, **kw):
        dsets = [slice_nldas_forcing_a(date, X_time_steps=max_time_steps)
                 for date in dates[:1]]
        feats = [dset.to_features().features for dset in dsets]
        return MLDataset(OrderedDict([('features', xr.concat(feats))]))
    fit_transform = transform


max_time_steps = DEFAULT_MAX_STEPS // 2
date = START_DATE
dates = np.array([START_DATE - datetime.timedelta(hours=hr)
                 for hr in range(max_time_steps)])

if __name__ == "__main__":

    pipe = Pipeline([
        ('time', Differencing(layers=FEATURE_LAYERS)),
        ('flatten', Flatten()),
        ('soil_phys', AddSoilPhysicalChemical(soils_dset=read_nldas_soils())),
        ('get_y', GetY(SOIL_MOISTURE)),
        ('log', preprocessing.FunctionTransformer(func=log_trans_only_positive)),
        ('scaler', preprocessing.MinMaxScaler(feature_range=(1e-2, 1e-2 + 1))),
        ('pca', ChooseWithPreproc()),
        ('estimator', linear_model.LinearRegression(n_jobs=-1)),
    ])

    ea = EaSearchCV(pipe,
                    n_iter=10,
                    param_distributions=param_distributions,
                    sampler=Sampler(),
                    ngen=NGEN,
                    model_selection=model_selection,
                    scheduler=None,
                    refit=True,
                    refit_Xy=Sampler().fit_transform([START_DATE]),
                    cv=KFold(3))
    download_data()
    ea.fit(dates)
'''
date += ONE_HR
current_file = get_file_name('fit_model', date)

dump(ea, tag, date)
estimators.append(ea)
second_layer = MultiLayer(estimator=linear_model.LinearRegression,
                          estimators=estimators)
second_layer.fit(X)
pred = ea.predict(X)
'''