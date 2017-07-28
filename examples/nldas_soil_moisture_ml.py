from __future__ import print_function

from collections import OrderedDict
import datetime
from functools import partial
import os

import dill
from earthio import Canvas, drop_na_rows, flatten
from elm.pipeline import Pipeline, steps
from elm.pipeline.ensemble import ensemble
from elm.pipeline.predict_many import predict_many
from pydap.cas.urs import setup_session
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (LinearRegression, SGDRegressor,
                                  RidgeCV, Ridge)
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from elm.model_selection.sorting import pareto_front
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

VIC, FORA = ('NLDAS_VIC0125_H', 'NLDAS_FORA0125_H',)

NGEN = 1
NSTEPS = 1

X_TIME_STEPS = 144
X_TIME_AVERAGING = [0, 3, 6, 9, 12, 18, 24, 36, 48] + list(range(72, X_TIME_STEPS, 24))

BASE_URL = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/{}/{:04d}/{:03d}/{}'

SOIL_MOISTURE = 'SOIL_M_110_DBLY'

PREDICTOR_COLS = None # Set this to a list to use only a subset of FORA DataArrays

START_DATE = datetime.datetime(2000, 1, 1, 1, 0, 0)

def get_session():
    u, p = os.environ['NLDAS_USER'], os.environ['NLDAS_PASS']
    return setup_session(u, p)

SESSION = get_session()

np.random.seed(42)  # TODO remove

TOP_N_MODELS = 6
MIN_MOISTURE_BOUND, MAX_MOISTURE_BOUND = -80, 2000
MIN_R2 = 0.

DIFFERENCE_COLS = [  # FORA DataArray's that may be differenced
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

def make_url(year, month, day, hour, dset, nldas_ver='002'):
    '''For given date components, data set identifier,
    and NLDAS version, return URL and relative path for a file

    Returns:
        url: URL on hydro1.gesdisc.eosdis.nasa.gov
        rel: Relative path named like URL pattern
    '''
    start = datetime.datetime(year, 1, 1)
    actual = datetime.datetime(year, month, day)
    julian = int(((actual - start).total_seconds() / 86400) + 1)
    vic_ver = '{}.{}'.format(dset, nldas_ver)
    fname_pat = '{}.A{:04d}{:02d}{:02d}.{:04d}.{}.grb'.format(dset, year, month, day, hour * 100, nldas_ver)
    url = BASE_URL.format(vic_ver, year, julian, fname_pat)
    rel = os.path.join('{:04d}'.format(year),
                       '{:03d}'.format(julian),
                       fname_pat)
    return url, rel


def get_file(*args, **kw):
    '''Pass date components and dset arguments to make_url and
    download the file if needed.  Return the relative path
    in either case

    Parameters:
        See make_url function above: Arguments are passed to that function

    Returns:
        rel:  Relative path
    '''
    url, rel = make_url(*args, **kw)
    path, basename = os.path.split(rel)
    if not os.path.exists(rel):
        if not os.path.exists(path):
            os.makedirs(path)
        print('Downloading', url, 'to', rel)
        r = SESSION.get(url)
        with open(rel, 'wb') as f:
            f.write(r.content)
    return rel


def get_nldas_fora_X_and_vic_y(year, month, day, hour,
                           vic_or_fora, band_order=None,
                           prefix=None, data_arrs=None,
                           keep_columns=None):
    '''Load data from VIC for NLDAS Forcing A Grib files

    Parameters:
        year: year of forecast time
        month: month of forecast time
        day: day of forecast time
        vic_or_fora: string indicating which NLDAS data source
        band_order: list of DataArray names already loaded
        prefix: add a prefix to the DataArray name from Grib
        data_arrs: Add the DataArrays to an existing dict
        keep_columns: Retain only the DataArrays in this list, if given
    Returns:
        tuple or (data_arrs, band_order) where data_arrs is
        an OrderedDict of DataArrays and band_order is their
        order when they are flattened from rasters to a single
        2-D matrix
    '''
    data_arrs = data_arrs or OrderedDict()
    band_order = band_order or []
    path = get_file(year, month, day, hour, dset=vic_or_fora)
    dset = xr.open_dataset(path, engine='pynio')
    for k in dset.data_vars:
        if keep_columns and k not in keep_columns:
            continue
        arr = getattr(dset, k)
        if sorted(arr.dims) != ['lat_110', 'lon_110']:
            continue
        #print('Model: ',f, 'Param:', k, 'Detail:', arr.long_name)
        lon, lat = arr.lon_110, arr.lat_110
        geo_transform = [lon.Lo1, lon.Di, 0.0,
                         lat.La1, 0.0, lat.Dj]
        shp = arr.shape
        canvas = Canvas(geo_transform, shp[1], shp[0], arr.dims)
        arr.attrs['canvas'] = canvas
        if prefix:
            band_name = '{}_{}'.format(prefix, k)
        else:
            band_name = k
        data_arrs[band_name] = arr
        band_order.append(band_name)
    return data_arrs, band_order


def sampler(date, X_time_steps=144, **kw):
    '''Sample the NLDAS Forcing A GriB file(s) for X_time_steps
    and get a VIC data array from GriB for the current step to use
    as Y data

    Parameters:
        date: Datetime object on an integer hour - VIC and FORA are
              retrieved for this date
        X_time_steps: Number of preceding hours to include in sample
        **kw:  Ignored

    Returns:
        this_hour_data: xarray.Dataset
    '''
    year, month, day, hour = date.year, date.month, date.day, date.hour
    data_arrs = OrderedDict()
    band_order = []
    forecast_time = datetime.datetime(year, month, day, hour, 0, 0)
    data_arrs, band_order = get_nldas_fora_X_and_vic_y(year, month,
                                                   day, hour,
                                                   VIC, band_order=band_order,
                                                   prefix=None,
                                                   data_arrs=data_arrs,
                                                   keep_columns=[SOIL_MOISTURE])
    for hours_ago in range(X_time_steps):
        file_time = forecast_time - datetime.timedelta(hours=hours_ago)
        y, m = file_time.year, file_time.month
        d, h = file_time.day, file_time.hour
        data_arrs, band_order = get_nldas_fora_X_and_vic_y(y, m,
                                                       d, h,
                                                       FORA,
                                                       band_order=band_order,
                                                       prefix='hr_{}'.format(hours_ago),
                                                       data_arrs=data_arrs,
                                                       keep_columns=PREDICTOR_COLS)
    attrs = dict(band_order=band_order)
    return xr.Dataset(data_arrs, attrs=attrs)


def get_y(y_field, X, y=None, sample_weight=None, **kw):
    '''Get the VIC Y column out of a flattened Dataset
    of FORA and VIC DataArrays'''
    assert ('flat',) == tuple(X.data_vars)
    y = X.flat[:, X.flat.band == y_field].values
    flat = X.flat[:, X.flat.band != y_field]
    X2 = xr.Dataset({'flat': flat}, attrs=X.attrs)
    X2.attrs['canvas'] = X.flat.canvas
    X2.attrs['band_order'].remove(y_field)
    return X2, y, sample_weight


def r_squared_mse(y_true, y_pred, sample_weight=None, multioutput=None):

    r2 = r2_score(y_true, y_pred,
                  sample_weight=sample_weight, multioutput=multioutput)
    mse = mean_squared_error(y_true, y_pred,
                             sample_weight=sample_weight,
                             multioutput=multioutput)
    bounds_check = np.min(y_pred) > MIN_MOISTURE_BOUND
    bounds_check = bounds_check&(np.max(y_pred) < MAX_MOISTURE_BOUND)
    print('Scoring - std', np.std(y_true), np.std(y_pred))
    print('Scoring - median', np.median(y_true), np.median(y_pred))
    print('Scoring - min', np.min(y_true), np.min(y_pred))
    print('Scoring - max', np.max(y_true), np.max(y_pred))
    print('Scoring - mean', np.mean(y_true), np.mean(y_pred))
    print('Scoring - MSE, R2, bounds', mse, r2, bounds_check)
    return (float(mse),
            float(r2),
            int(bounds_check))


def ensemble_init_func(pipe, **kw):
    '''Create an ensemble of regression models to predict soil moisture
    where PCA, scaling, and/or log transformation may follow preamble
    steps of flattening a Dataset and extracting the Y data, among other
    preprocessors.

    Parameters:
        pipe: Ignored
        **kw: Keyword arguments:
            scalers: List of (name, scaler) tuples such as
                     [('StandardScaler', steps.StandardScaler(with_mean=True)),
                      ('RobustScaler', steps.RobustScaler(with_centering=True))]
            n_components: List of PCA # of components to try. May include None
                          if skipping PCA step
            estimators: List of (name, estimator) tuples where estimator
                        may be any scikit-learn-like regressor, e.g.
                        [('estimator', LinearRegression())]
            log:        Log transform step, e.g.:
                        ('log', steps.ModifySample(log_scaler))
            summary:    String summary of premable steps to prepend to
                        parameter summary

    Returns:
        ensemble: List of Pipeline instances
    '''
    ensemble = []
    scalers = kw['scalers']
    n_components = kw['n_components']
    pca = kw['pca']
    estimators = kw['estimators']
    preamble = kw['preamble']
    summary_template = kw['summary']
    minmax_bounds = kw['minmax_bounds']
    log = kw['log']

    for s_label_0, scale_0 in scalers:
        if 'MinMax' in s_label_0:
            # Make MinMaxScaler objects
            labels = [s_label_0 + repr(mb) for mb in minmax_bounds]
            scalers_with_params = [scale_0(*mb) for mb in minmax_bounds]
            scalers_with_params = zip(labels, scalers_with_params)
        elif scale_0:
            # Just keep the StandardScaler as is
            scalers_with_params = [(s_label_0, scale_0())]
        else:
            # No scaling
            scalers_with_params = [(s_label_0, None)]
        for s_label, scale in scalers_with_params:
            for n_c in n_components:
                for e_label, estimator in estimators:
                    scale_step = [scale] if scale else []
                    if 'MinMax' in s_label:
                        # Log transform only works with MinMaxScaler
                        # and positive min bound
                        scale_step += [log]
                    pca_step = [pca()] if n_c and scale else []
                    new = Pipeline(preamble() +
                                   scale_step +
                                   pca_step +
                                   [estimator()],
                                   **pipeline_kw)
                    if pca_step:
                        new.set_params(pca__n_components=n_c)
                        msg = '{} components'.format(n_c)
                    else:
                        msg = ' (None)'
                    args = (s_label, msg, e_label)
                    summary = ': Scaler: {} PCA: {} Estimator: {}'.format(*args)
                    new.summary = summary_template + summary
                    print(new.summary)
                    ensemble.append(new)
    return ensemble


_last_idx = 0
def next_tag():
    '''Make a tag for a model'''
    global _last_idx
    _last_idx += 1
    return 'new_member_{}'.format(_last_idx)


def model_selection(ensemble, **kw):
    '''Pareto sort the ensemble by objective scores, keeping
    TOP_N_MODELS best models and initializing new models
    to keep the ensemble size constant.'''

    # Get the MSE and R2 scores
    scores = np.array([model._score[:-1] for _, model in ensemble])
    # Minimization/maximization weights for MSE and R2 scores
    wts = [-1, 1]
    # Sort by Pareto optimality on MSE, R2 scores
    ensemble = [ensemble[idx] for idx in pareto_front(wts, scores)]
    # Apply some bounds checks:
        # 1) R2 > 0.3 and
        # 2) Minimum predicted soil moisture > -10
    ensemble = [(tag, model) for tag, model in ensemble
                if model._score[1] > MIN_R2 # min R**2 criterion
                and model._score[2]]        # mostly postive criterion (moisture)
                                            # and less than max possible
    print('Scores:', [model._score for _, model in ensemble])
    last_gen = kw['ngen'] - 1 == kw['generation']
    if last_gen:
        return ensemble[:TOP_N_MODELS]
    new = kw['ensemble_init_func'](None)
    np.random.shuffle(new)
    new = [(next_tag(), model) for model in new]
    np.random.shuffle(new)
    return ensemble[:TOP_N_MODELS] + new[:len(ensemble) - TOP_N_MODELS]


def second_layer_input_matrix(X, models):
    '''Build a second layer model input matrix by taking the
    metadata from X given to the first layer models and forming
    a new matrix from the 1-D predictions of the first layer models
    '''
    preds = predict_many(dict(X=X), to_raster=False,
                         ensemble=models)
    example = preds[0].flat
    input_matrix = np.empty((example.shape[0], len(preds)))
    for j, pred in enumerate(preds):
        input_matrix[:, j] = pred.flat.values[:, 0]
    attrs = X.attrs.copy()
    attrs['old_dims'] = [X[SOIL_MOISTURE].dims] * len(preds)
    attrs['canvas'] = X[SOIL_MOISTURE].canvas
    tags = [tag for tag, _ in models]
    arr = xr.DataArray(input_matrix,
                       coords=[('space', example.space),
                               ('band', tags)],
                       dims=('space', 'band'),
                       attrs=attrs)
    return xr.Dataset(dict(flat=arr), attrs=attrs)


def ensemble_layer_2(pipe, **kw):
    '''A simple model for the second layer (model on models).
    RidgeCV is a good choice in the second layer since
    colinearity is expected among the predictions from the
    first layer that form an input matrix to the second layer'''
    return [Pipeline([RidgeCV()], **pipeline_kw)]


def train_model_on_models(last_hour_data, this_hour_data, init_func):
    '''Given input NLDAS FORA data from last hour and this hour,
    train on the last hour and use the trained models to predict
    the current hour's soil moisture

    Parameters:

        last_hour_data: Dataset from sampler() function above
        this_hour_data: Dataset from sampler() function above, typically
                        one hour later than last_hour_data
        init_func:      Partial of ensemble_init_func that can
                        be passed to the training function "ensemble"

    Returns:
        last_hour_data: See above
        this_hour_data: See above
        models:         First layer trained Pipelines on last_hour_data
        preds:          First layer predictions from "models" on this_hour_data
        models2:        Second layer trained Pipelines on last_hour_data
        preds2:         Second layer predictions from "models2" on this_hour_data

    '''
    for hour in ('last', 'this'):
        if hour == 'last':
            X = last_hour_data
        else:
            X = this_hour_data
        X_clean, true_y, _ = get_y(SOIL_MOISTURE,
                                   drop_na_rows(flatten(X)))
        if hour == 'last':
            models = ensemble(None, ngen=NGEN, X=X,
                              ensemble_init_func=init_func,
                              model_selection=model_selection,
                              model_selection_kwargs=dict(ensemble_init_func=init_func))
        else:
            preds = predict_many(dict(X=X),
                                 ensemble=models)
        X_second = second_layer_input_matrix(X, models)
        X_second.attrs['drop_na_rows'] = X_clean.drop_na_rows
        X_second.attrs['shape_before_drop_na_rows'] = X_clean.shape_before_drop_na_rows
        if hour == 'last':
            models2 = ensemble(None, ngen=1,
                               X=X_second, y=true_y,
                               ensemble_init_func=ensemble_layer_2)
        else:
            preds2 = predict_many(dict(X=X_second),
                                  ensemble=models2)
    return last_hour_data, this_hour_data, models, preds, models2, preds2


def avg_arrs(*arrs):
    '''Take the mean of a variable number of xarray.DataArray objects and
    keep metadata from the first DataArray given'''
    s = arrs[0]
    if len(arrs) > 1:
        for a in arrs[1:]:
            s += a
    s = s / float(len(arrs))
    s.attrs.update(arrs[0].attrs)
    return s


def differencing_integrating(X, y=None, sample_weight=None, **kw):

    X_time_steps = kw['X_time_steps']
    difference_cols = kw['difference_cols']
    X_time_averaging = kw['X_time_averaging']
    X = X.copy(deep=True)
    X.attrs['band_order'] = X.band_order[:]
    new_X = OrderedDict([(k, getattr(X, k)) for k in X.data_vars
                          if k.startswith('hr_0_') or SOIL_MOISTURE == k])

    assert len(X.data_vars) == len(X.band_order), repr((len(X.data_vars), len(X.band_order)))
    band_order = list(new_X)
    running_fields = []
    running_diffs = []
    last_hr = 0
    for col in difference_cols:
        for first_hr, second_hr in zip(X_time_averaging[:-1],
                                       X_time_averaging[1:]):
            for i in range(first_hr, second_hr):
                old = 'hr_{}_{}'.format(first_hr, col)
                new = 'hr_{}_{}'.format(second_hr, col)
                old_array = X.data_vars[old]
                new_array = X.data_vars[new]
                running_fields.append(old_array)
                diff = new_array - old_array
                diff.attrs.update(new_array.attrs.copy())
                running_diffs.append(diff)
            diff_col_name = 'diff_{}_{}_{}'.format(first_hr, second_hr, col)
            new_X[diff_col_name] = avg_arrs(*running_diffs)
            running_diffs = []
            new_X[new] = avg_arrs(*running_fields)
            running_fields = []
            band_order.extend((diff_col_name, old))
    X = xr.Dataset(new_X, attrs=X.attrs)
    X.attrs['band_order'] = band_order
    assert len(X.data_vars) == len(X.band_order), repr((len(X.data_vars), len(X.band_order)))
    return X, y, sample_weight


def log_scaler(X, y=None, sample_weight=None, **kw):
    Xnew = OrderedDict()
    for j in range(X.flat.shape[1]):
        minn = X.flat[:, j].min().values
        if minn <= 0:
            continue
        X.flat.values[:, j] = np.log10(X.flat.values[:, j])
    return X, y, sample_weight


def add_sample_weight(X, y=None, sample_weight=None, **kw):
    '''Modify this function to return a sample_weight
    if needed.  sample_weight returned should be a 1-D
    NumPy array.  Currently it is weighting the pos/neg deviations.
    '''
    sample_weight = np.abs((y - y.mean()) / y.std())
    return X, y, sample_weight


pipeline_kw = dict(scoring=make_scorer(r_squared_mse))
flat_step = ('flatten', steps.Flatten())
drop_na_step = ('drop_null', steps.DropNaRows())
kw = dict(X_time_steps=X_TIME_STEPS,
          X_time_averaging=X_TIME_AVERAGING,
          difference_cols=DIFFERENCE_COLS)

diff_in_time = ('diff', steps.ModifySample(differencing_integrating, **kw))
get_y_step = ('get_y', steps.ModifySample(partial(get_y, SOIL_MOISTURE)))
robust = lambda: ('normalize', steps.RobustScaler(with_centering=False))
standard = lambda: ('normalize', steps.StandardScaler(with_mean=False))
minmax = lambda minn, maxx: ('minmax',
                             steps.MinMaxScaler(feature_range=(minn, maxx)))
minmax_bounds = [(0.01, 1.01), (0.05, 1.05),
                 (0.1, 1.1), (0.2, 1.2),  (1, 2),]
weights = ('weights', steps.ModifySample(add_sample_weight))
log = ('log', steps.ModifySample(log_scaler))
preamble = lambda: [diff_in_time,
                    flat_step,
                    drop_na_step,
                    get_y_step,
                    weights,]

linear = lambda: ('estimator', LinearRegression(n_jobs=-1))
pca = lambda: ('pca', steps.Transform(PCA()))
n_components = [None, 4, 6, 8, 10]

def main():
    '''
    Beginning on START_DATE, step forward hourly, training on last
    hour's NLDAS FORA dataset with transformers in a 2-layer hierarchical
    ensemble, training on the last hour of data and making
    out-of-training-sample predictions for the current hour.  Makes
    a dill dump file for each hour run. Runs fro NSTEPS hour steps.
    '''
    date = START_DATE
    add_hour = datetime.timedelta(hours=1)
    get_file_name = lambda date: date.isoformat(
                        ).replace(':','_').replace('-','_') + '.dill'
    scalers = zip(('MinMaxScaler', 'RobustScaler', 'StandardScaler', 'None'),
                  (minmax, robust, standard, None))
    estimators = zip(('LinearRegression', ),
                     (linear, ))
    init_func = partial(ensemble_init_func,
                        pca=pca,
                        scalers=scalers,
                        n_components=n_components,
                        estimators=estimators,
                        preamble=preamble,
                        log=log,
                        minmax_bounds=minmax_bounds,
                        summary='Flatten, Subset, Drop NaN Rows, Get Y Data, Difference X in Time')
    for step in range(NSTEPS):
        last_hour_data = sampler(date, X_time_steps=X_TIME_STEPS)
        date += add_hour
        this_hour_data = sampler(date, X_time_steps=X_TIME_STEPS)
        current_file = get_file_name(date)
        out = train_model_on_models(last_hour_data, this_hour_data, init_func)
        dill.dump(out, open(current_file, 'wb'))
        print('Dumped to:', current_file)
        l2, t2, models, preds, models2, preds2 = out
        layer_1_scores = [model._score for _, model in models]
        layer_2_scores = [model._score for _, model in models2]
        print('Scores in layer 1 models:', layer_1_scores)
        print('Scores in layer 2 models:', layer_2_scores)
    return last_hour_data, this_hour_data, models, preds, models2, preds2

if __name__ == '__main__':
    last_hour_data, this_hour_data, models, preds, models2, preds2 = main()

