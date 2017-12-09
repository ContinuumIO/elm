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
                                FEATURE_LAYERS_CHOICES,
                                SOIL_MOISTURE)
from read_nldas_soils import (download_data, read_nldas_soils,
                              flatten_horizons,
                              SOIL_FEAUTURES_CHOICES)
from ts_raster_steps import differencing_integrating
from changing_structure import ChooseWithPreproc
import xarray as xr

NGEN = 3
NSTEPS = 1
WATER_MASK = -9999
DEFAULT_CV = 3
DEFAULT_MAX_STEPS = 144

START_DATE = datetime.datetime(2000, 1, 1, 1, 0, 0)

print('nldas_soil_features')
SOIL_PHYS_CHEM = flatten_horizons(read_nldas_soils())
print('post_features')
ONE_HR = datetime.timedelta(hours=1)
TIME_OPERATIONS = ('mean',
                   'std',
                   'sum',
                   ('diff', 'mean'),
                   ('diff', 'std'),
                   ('diff', 'sum'))
REDUCERS = [('mean', x) for x in TIME_OPERATIONS if x != 'mean']

np.random.seed(42)  # TODO remove

class LogOnlyPositive(Step):
    use_transform = False
    def transform(self, X, y=None, **kw):
        print('LOP,', X, y)
        if len(X) == 2 and isinstance(X, tuple):
            X, y = X
        assert y is not None
        if not self.get_params()['use_transform']:
            return X, y
        for j in range(X.shape[1]):
            minn = X[:, j].min()
            if minn <= 0:
                continue
            X[:, j] = np.log10(X[:, j])
        return X, y
    fit_transform = transform


class AddSoilPhysicalChemical(Step):
    add = True
    subset = None

    def transform(self, X, y=None, **kw):
        global SOIL_PHYS_CHEM
        if not self.add:
            return X
        soils = SOIL_PHYS_CHEM.copy()
        if self.subset:
            choices = SOIL_FEAUTURES_CHOICES[self.subset]
            soils = OrderedDict([(layer, arr)
                                 for layer, arr in soils.data_vars.items()
                                 if layer in choices])
            soils = MLDataset(soils)
        renamed = X.rename(dict(lon_110='x', lat_110='y'))
        return soils.reindex_like(renamed, method='nearest').merge(renamed, compat='broadcast_equals')

    fit_transform = transform

param_distributions = {
    'log__use_transform': [False],
    'scaler__feature_range': [(x, x * 2) for x in np.linspace(0, 1, 10)],
    'time__hours_back': [144],
    'time__weight_a': [1, 2, 5, 10],
    'time__weight_b': [0, 0.5],
    'subset__include': FEATURE_LAYERS_CHOICES[-2:],
    'soil_phys__add': [False],
    'soil_phys__subset': tuple(x for x in SOIL_FEAUTURES_CHOICES if 'MOSAIC' not in x),
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
    'early_stop': None,
}

def get_file_name(tag, date):
    date = date.isoformat().replace(':','_').replace('-','_')
    return '{}-{}.dill'.format(tag, date)


def dump(obj, tag, date):
    fname = get_file_name(tag, date)
    return getattr(obj, 'dump', getattr(obj, 'to_netcdf'))(fname)


class TimeAvg(Step):
    hours_back = 144
    weight_a = 1.0
    weight_b = 0.   # a, b = 1, 0 is uniform weighting (ax + b where x is hours)
    reducer = 'sum'
    reduce_dim = 'time'

    def transform(self, X, y=None, **kw):
        dset = OrderedDict()
        a, b = self.weight_a, self.weight_b
        if a is None:
            a = 1.
        if b is None:
            b = 0.
        weights = None
        for layer, arr in X.data_vars.items():
            if not 'time' in arr.dims:
                continue
            tidx = arr.dims.index('time')
            siz = [1] * len(arr.dims)
            siz[tidx] = arr.time.values.size
            if weights is None:
                time = np.linspace(0, siz[tidx], siz[tidx])
                weights = a * time + b
                weights /= weights.sum()
                weights.resize(tuple(siz))

            if arr.source == 'FORA':
                weighted = (arr * weights)
                reducer_func = getattr(weighted, self.reducer)
                dset[layer] = reducer_func(dim=self.reduce_dim)
            else:
                dset[layer] = arr.isel(time=siz[tidx] - 1)
        print('dset tt', dset)
        return MLDataset(dset)

    fit_transform = transform


class ChooseFeatures(Step):
    include = None
    exclude = None
    def transform(self, X, y=None, **kw):
        subset = OrderedDict()
        include = list(self.include or X.data_vars)
        if SOIL_MOISTURE not in include:
            include += [SOIL_MOISTURE]
        for layer, arr in X.data_vars.items():
            if layer in include:
                if self.exclude and layer in self.exclude:
                    continue
                subset[layer] = arr
        return MLDataset(subset)

    fit_transform = transform


class Sampler(Step):

    hours_back = DEFAULT_MAX_STEPS

    def transform(self, dates, y=None, **kw):
        print('Sampler Called')
        dset = slice_nldas_forcing_a(dates[0],
                                     hours_back=self.hours_back)
        return dset

    fit_transform = transform


max_time_steps = DEFAULT_MAX_STEPS // 2
date = START_DATE
dates = np.array([START_DATE - datetime.timedelta(hours=hr)
                 for hr in range(max_time_steps)])
time_avg = TimeAvg()
soil_phys = AddSoilPhysicalChemical(subset='COS_STEX')
feature_select = ChooseFeatures(include=FEATURE_LAYERS_CHOICES[-1])
get_y = GetY(column=SOIL_MOISTURE)
scaler = preprocessing.MinMaxScaler(feature_range=(1e-2, 1e-2 + 1))
log = LogOnlyPositive(use_transform=True)
pca = decomposition.PCA()
reg = linear_model.LinearRegression(n_jobs=-1)
pipe = Pipeline([
    ('subset', feature_select),
    ('time', time_avg),
    ('soil_phys', soil_phys),
    ('get_y', get_y),
    ('scaler', scaler),
    ('log', log),
    #('pca', pca),
    ('estimator', reg),
])

sampler = Sampler()
X = sampler.fit_transform(dates[:1])
outputs = [(X,)]
y = None
for label, step in pipe.steps[:-1]:
    out = step.fit_transform(X, y=y)
    if len(out) == 2 and isinstance(out, tuple):
        X, y = out
        was_2 = True
    else:
        X = out
        was_2 = False
    outputs.append((type(X), type(y), label, X, y))
    print('output for label {} is of type {} and {} - {}'.format(label, type(X), type(y), was_2))

fitted = pipe.steps[-1][1].fit(X, y=y)



ea = EaSearchCV(pipe,
                n_iter=4,
                param_distributions=param_distributions,
                sampler=sampler,
                ngen=2,
                model_selection=model_selection,
                scheduler=None,
                refit=True,
                refit_Xy=sampler.fit_transform([START_DATE]),
                cv=KFold(3))



def main():
    print('Download')
    download_data()
    print('Downloaded')
    print('Fit')
    fitted = ea.fit(dates)
    print('Done')
    return fitted


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ea = main()
        ea.dump('soil_moisture_regression_saved.pkl')
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