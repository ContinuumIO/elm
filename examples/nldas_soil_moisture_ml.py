
from collections import OrderedDict
import datetime
from functools import partial
import os

import dill
from elm.pipeline import Pipeline, steps
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression
from xarray_filters.pipeline import Step
from xarray_filters import MLDataset
from xarray_filters.ts_grid_tools import TSProbs

from sklearn.metrics import (mean_squared_log_error,
                             make_scorer,
                             mean_squared_error,
                             mean_absolute_error,
                             explained_variance_score,
                             r2_score)
from elm.mldataset.util import _split_transformer_result
from elm.model_selection import EaSearchCV
from elm.pipeline import Pipeline
from elm.pipeline.steps.linear_model import LinearRegression as LR
from elm.pipeline.steps.linear_model import SGDRegressor as SGDR
from elm.pipeline.steps.feature_selection import SelectPercentile
from elm.pipeline.steps.preprocessing import (PolynomialFeatures,
                                              MinMaxScaler)

from read_nldas_soils import (read_nldas_soils,
                              soils_join_forcing,
                              download_data)
from read_nldas_forcing import (extract_soil_moisture_column,
                                SOIL_MOISTURE,
                                slice_nldas_forcing_a,)


DEFAULT_HOURS = 144
DATE = datetime.datetime(2000, 1, 1)
SOIL = None
if __name__ == '__main__':
    SOIL = read_nldas_soils()


def get_soil(X, y=None, subset=None, **kw):
    global SOIL
    if SOIL is None:
        SOIL = read_nldas_soils()
    return soils_join_forcing(SOIL, X, subset=subset)


class GetSoil(Step):
    subset = 'COS_RAWL'
    def transform(self, X, y=None, **kw):
        X, y = _split_transformer_result(X, y)
        return get_soil(X, y, self.subset, **kw)
    fit_transform = transform


class LogTrans(Step):
    use_log = True
    def transform(self, X, y=None, **kw):
        X, y = _split_transformer_result(X, y)
        if not self.use_log:
            return X, y
        X2 = X.copy()
        X2[X2 > 0.] = np.log10(X2[X2 > 0.])
        return X2, y


def get_bins(b, max_t):
    log_hrs = np.logspace(np.log10(DEFAULT_HOURS), 0, b)
    return np.unique(max_t - log_hrs.astype(np.int32))


def time_series_agg(arr, **kw):
    bins = get_bins(kw['bins'], arr.time.values.max())
    t = np.sort(arr.time.values)
    for (start, end) in zip(bins[:-1], bins[1:]):
        avg_time_bin = arr.isel(time=range(start, end)).mean(dim='time')
        yield start, end, avg_time_bin


class TimePDFPlusRecent(Step):
    include_latest = True
    bins = DEFAULT_HOURS // 2
    def transform(self, X, y=None, **kw):
        Xnew = OrderedDict()
        p = self.get_params()
        for layer, arr in X.data_vars.items():
            if layer == SOIL_MOISTURE:
                Xnew[layer] = arr.sel(time=arr.time.values.max())
                continue
            for start, end, arr in time_series_agg(arr, **p):
                Xnew['{}_{}_{}'.format(layer, start, end)] = arr
        return MLDataset(Xnew)


class GetY(Step):
    def transform(self, X, y=None, **kw):
        return extract_soil_moisture_column(X, y=y,
                                            column=SOIL_MOISTURE,
                                            **kw)

def weight_y_resids(y):
    return np.abs(y - y.mean()) / y.std()


def calc_sample_weight(cls):
    '''Class decorator to wrap a "fit" method,
    creating a sample weight that favors fitting
    minima/maxima'''
    cls._old_fit = cls.fit
    def fit_new(self, X, y, **kw):
        kw['sample_weight'] = weight_y_resids(y)
        return self._old_fit(X, y, **kw)
    cls.fit = fit_new
    return cls


LinearRegression = calc_sample_weight(LR)
ols = [LinearRegression(n_jobs=-1, fit_intercept=f, normalize=n)
       for f in (True, False)
       for n in (True, False)]
SGDRegressor = calc_sample_weight(SGDR)
sgd = [SGDRegressor(penalty=p, alpha=a)
        for p in ('l1', 'l2')
        for a in np.logspace(-4, 2)]

estimators = ols + sgd
param_distributions = {
    'log__use_log': [True, False],
    'scaler__feature_range': [(0.01, 1.01),
                              (0.05, 1.05),
                              (0.1, 1.1),
                              (0.2, 1.2),
                              (0.3, 1.3),
                              (0.5, 1.5),
                              (1, 2),],
    'poly__interaction_only': [True, False],
    'selector__percentile': np.linspace(10, 90, 10),
    'est': estimators,
}
model_selection = dict(
    k=16,
    mu=24,
    cxpb=0.4,
    indpb=0.5,
    mutpb=0.9,
    eta=20,
    param_grid_name='param_grid',
    select_method='selNSGA2',
    crossover_method='cxTwoPoint',
    mutate_method='mutUniformInt',
    init_pop='random',
    early_stop=None,
    toolbox=None
)


def mean_4th_power_error(y_true, y_pred,
                         sample_weight=None,
                         **kw):
    '''4th power error penalizes the errors in minima/maxima'''
    if sample_weight is None:
        sample_weight = 1.
    weighted_resids = (y_true * sample_weight - y_pred * sample_weight)
    return (weighted_resids ** 4).mean()


def fit_once(pdf_params=None, soil_params=None):
    pdf_params = pdf_params or {}
    soil_params = soil_params or {}
    dset = slice_nldas_forcing_a(DATE,
                                 hours_back=DEFAULT_HOURS)
    time_binning = TimePDFPlusRecent(**pdf_params)
    feat = time_binning.fit_transform(dset)
    X, y = GetY().fit_transform(feat)
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('log', LogTrans()),
        ('poly', PolynomialFeatures(degree=1)),
        ('selector', SelectPercentile(f_regression, 50)),
        ('est', SGDRegressor())])
    ea = EaSearchCV(pipe,
                    n_iter=model_selection['mu'],
                    score_weights=[1],
                    scoring=make_scorer(mean_4th_power_error),
                    param_distributions=param_distributions,
                    ngen=8,
                    model_selection=model_selection,
                    cv=KFold(5),
                    refit=True)
    ea.fit(X, y)
    pred = ea.predict(X)
    df = pd.DataFrame(ea.cv_results_)
    return feat, X, y, pipe, ea, pred, df


def all_regression_metrics(y_true, y_pred, sample_weight=None):
    scorers = (
        mean_squared_log_error,
        mean_squared_error,
        mean_absolute_error,
        explained_variance_score,
        r2_score,
        mean_4th_power_error,
    )
    names = (
        'mean_squared_log_error',
        'mean_squared_error',
        'mean_absolute_error',
        'explained_variance_score',
        'r2_score',
        'mean_4th_power_error',
    )
    sample_weight = weight_y_resids(y_true)
    args = (y_true, y_pred,)
    out = {k: s(*args, sample_weight=sample_weight)
           for k, s in zip(names, scorers)}
    print('Scores:', out)
    return out


if __name__ == '__main__':
    download_data() # Soils physical / class / texture
    pdf_params = dict(bins=50, include_latest=True)
    soil_params = dict(subset='COS_RAWL')
    feat, X, y, pipe, ea, pred, df = fit_once(pdf_params, soil_params)
    scores = all_regression_metrics(y, pred)
    with open('soil_moisture_results.pkl', 'wb') as f:
        dill.dump([feat, X, y, pipe, ea, pred, df, scores], f)

