
#!conda install -c conda-forge holoviews geoviews bokeh
import datetime
from collections import OrderedDict
import dask
import dask.array as da
from elm.model_selection import EaSearchCV
from elm.model_selection import CVCacheSampler
from elm.pipeline import Pipeline
from holoviews.operation import gridmatrix
import numpy as np
from sklearn.model_selection import KFold
from xarray_filters.datasets import make_blobs
from xarray_filters import MLDataset, for_each_array
from xarray_filters.pipeline import Step
from xarray_filters.reshape import concat_ml_features
import pandas as pd
import xarray as xr
import holoviews as hv
import parambokeh, param
import numba
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from bokeh.io import output_notebook
from sklearn import linear_model, cluster, decomposition, preprocessing
FREQ = '5Min'
PERIODS = 2000
START = np.datetime64('2018-03-01')


DEFAULTS = {'April_Weight': 0.7,
 'August_Weight': 0.26,
 'Conductivity_Convolution_Steps': 3,
 'Conductivity_Convolution_Weight1': 2.0,
 'Conductivity_Convolution_Weight2': 3.0,
 'Conductivity_Correlation_With_Flow': -0.4,
 'Conductivity_Correlation_With_Temperature': 0.6,
 'Conductivity_High_Flow_Mean': 30.0,
 'Conductivity_High_Flow_Stdev': 30.0,
 'Conductivity_Low_Flow_Mean': 120.0,
 'Conductivity_Low_Flow_Stdev': 15.0,
 'December_Weight': 1.91,
 'February_Weight': 1.3,
 'Flow_Convolution_Steps': 3,
 'Flow_Convolution_Weight1': 1.0,
 'Flow_Convolution_Weight2': 10.0,
 'Illicit_Discharge_Conductivity_Mean': 100,
 'Illicit_Discharge_Conductivity_Stdev': 40.0,
 'Illicit_Discharge_Peak_To_Base': 15,
 'Illicit_Discharge_Temperature_Mean': 52.0,
 'Illicit_Discharge_Temperature_Stdev': 8.0,
 'Illicit_Flow_Log10_Mean': 0.7,
 'Illicit_Flow_Log10_Sigma': 0.3,
 'January_Weight': 1.8,
 'July_Weight': 0.22,
 'June_Weight': 0.5,
 'Log10_Mean': 1.4,
 'Log10_Sigma': 0.7,
 'Low_Flow_Threshold_As_Percent_Of_Mean': 25,
 'March_Weight': 1.16,
 'May_Weight': 0.55,
 'November_Weight': 1.75,
 'October_Weight': 1.15,
 'Peak_Hours_Of_Illicit_Discharge': (3, 6),
 'Prob_Dry_To_Wet': 0.04,
 'Prob_Wet_To_Dry': 0.04,
 'September_Weight': 0.58,
 'Temperature_Convolution_Steps': 4,
 'Temperature_Convolution_Weight1': 1.0,
 'Temperature_Convolution_Weight2': 2.0,
 'Temperature_Correlation_With_Flow': -0.4,
 'Temperature_High_Flow_Mean': 50.0,
 'Temperature_High_Flow_Stdev': 8.0,
 'Temperature_Low_Flow_Mean': 42.0,
 'Temperature_Low_Flow_Stdev': 4.0,
 'flow_month': np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]),
 'name': 'Monthly',
 'periods': 4000,
 'time_step': '5Min'}



class Fuzzy:
    cv = 0.2
    base = None
    def __init__(self, *args):
        self.args = args
    def transform(self):
        args = []
        for a in self.args:
            if not isinstance(a, Fuzzy):
                if a is None:
                    raise ValueError()
                sign = -1 if np.any(a < 0.) else 1
                a = np.abs(a)
                rng = self.cv * (self.base or a) / 2.
                a = np.random.uniform(a - rng, a + rng)
                a *= sign
            args.append(a)
        return args


@numba.njit(nogil=True)
def markov_lognormal_flow(periods, transition, time_factor, mean, sigma, convolve_weights):
    is_wet = 0
    old = 0.
    p_dry_to_dry, p_dry_to_wet = transition[0, :]
    p_wet_to_dry, p_wet_to_wet = transition[1, :]
    output = np.zeros(periods, dtype=np.float32)
    innovations = np.zeros(convolve_weights.shape, dtype=np.float32)
    for idx in range(periods):
        rand = np.random.rand()
        if (is_wet and rand < p_wet_to_wet) or (not is_wet and rand < p_dry_to_wet):
            new = np.random.lognormal(mean, sigma) * time_factor[idx]
            is_wet = 1
        else:
            new = 0.
            is_wet = 0
        if idx >= innovations.size:
            innovations[:-1] = innovations[1:]
            innovations[-1] = new
        else:
            innovations[idx] = new
        if is_wet:
            new = np.sum(innovations[:idx + 1] * convolve_weights[:idx + 1])
        output[idx] = old = new
    return output


@numba.njit(nogil=True)
def normal_covariates(flow,
                      conductivity_means, conductivity_stdevs, cond_convolve_weights,
                      temperature_means, temperature_stdevs, temp_convolve_weights,
                      is_wet_divisor, corr):

    cond_limit = 0.01
    temp_limit = 32.
    rands = np.empty((flow.size, 2), dtype=np.float64) * np.NaN
    cutoff = flow[flow != 0.].mean() / is_wet_divisor
    threshold_idx = (flow > cutoff).astype(np.uint8)
    for idx in (0, 1):
        where = threshold_idx == idx
        subset = np.log10(flow[where] + cutoff)
        if subset.size == 0:
            continue
        cond_mean = conductivity_means[idx]
        cond_stdev = conductivity_stdevs[idx]
        temp_mean = temperature_means[idx]
        temp_stdev = temperature_stdevs[idx]
        for j in (0, 1):
            if j == 0:
                std = cond_stdev
            else:
                std = temp_stdev
            rands[where, j] = np.random.normal(0., std, subset.size)
        rands[where, :] = np.dot(np.column_stack((rands[where, :], subset)), corr)[:, :2] + np.array([cond_mean, temp_mean], dtype=np.float64)

    output = rands.copy()
    for wts in (0, 1):
        if wts == 0:
            weights = cond_convolve_weights
            limit = cond_limit
        else:
            weights = temp_convolve_weights
            limit = temp_limit
        for idx in range(1, output.shape[0]):
            start = idx - weights.size
            if start <= 0:
                start = 0
                w = weights[-idx:]
            else:
                w = weights
            slicer = slice(start, idx)
            out = np.sum(w * rands[slicer, wts])
            delta = out - limit
            if delta < 0:
                out = np.abs(delta) / 2.
            output[slicer, wts] = out
    return output


def conductivity_flow_temperature(flow_args, conductivity_args, temperature_args, is_wet_divisor, corr):
    flow = markov_lognormal_flow(*flow_args)
    args = tuple(map(np.array, tuple(conductivity_args) + tuple(temperature_args))) + (is_wet_divisor, corr)
    cond_temp = normal_covariates(flow, *args)
    return flow, cond_temp


class WaterSeries(Step):

    label = None
    flow_log_mean = None#1.4
    flow_log_sigma = None#0.7
    flow_weights = None#(0, 10, 4)
    conductivity_means = None#(100, 50)
    conductivity_stdevs =None# (10, 12)
    cond_convolve_weights = None#(1, 2, 3)
    temperature_means = None#(49, 42)
    temperature_stdevs = None#(5, 10)
    temp_convolve_weights = None#(1, 1.5, 3)
    is_wet_divisor = None#4
    corr_cond_temp = None#0.7
    corr_cond_flow = None#-0.8
    corr_temp_flow = None#-0.4
    prob_dry_to_wet = None#0.01
    prob_wet_to_dry = None#0.04
    periods = None
    time_step = None
    flow_month = None
    flow_hour = None
    def get_fuzzy_params(self):
        p = self.get_params()
        try:
            for k, v in p.items():
                if k in ('periods', 'time_step', 'label',):
                    p[k] = v
                    continue
                if isinstance(v, (tuple, list)):
                    v = np.array(v, dtype=np.float64)
                p[k] = Fuzzy(v).transform()[0]
            p['transition'] = self.get_transition(p)
        except:
            raise ValueError('Failed on {} {}'.format(k, v))

        return p

    def get_transition(self, p):
        return np.array([[1 - p['prob_dry_to_wet'], p['prob_dry_to_wet']],
                         [p['prob_wet_to_dry'], 1 - p['prob_wet_to_dry']]])

    def transform(self, *a, **kw):
        p = self.get_fuzzy_params()
        index = pd.DatetimeIndex(start=START,
                                 freq=p['time_step'],
                                 periods=p['periods'])
        month, hour = index.month, index.hour
        time_factor = p['flow_month'][month - 1] * p['flow_hour'][hour]
        f = (index.size, p['transition'], time_factor,
             p['flow_log_mean'], p['flow_log_sigma'], p['flow_weights'])
        c = p['conductivity_means'], p['conductivity_stdevs'], p['cond_convolve_weights']
        t = p['temperature_means'], p['temperature_stdevs'], p['temp_convolve_weights']
        corr = np.array([[1, p['corr_cond_temp'], p['corr_cond_flow']],
                 [p['corr_cond_temp'], 1., p['corr_temp_flow']],
                 [p['corr_cond_flow'], p['corr_temp_flow'], 1.]], dtype=np.float64)
        options = (p['is_wet_divisor'], corr)
        flow, cond_temp = conductivity_flow_temperature(f, c, t, *options)
        df = pd.DataFrame(np.c_[cond_temp, flow], columns=['cond', 'temp', 'flow'], index=index)
        return df, p


class Mix(Step):
    waters = None
    ids = None
    def transform(self, *a, **kw):
        drop_zeros_from = ['cond', 'temp']
        assert isinstance(self.waters, (tuple, list)) and len(self.waters) == 2
        dfs = []
        labels = []
        flows = []
        for w in self.waters:
            labels.append(w.label)
            df, p = w.transform()
            old_cols = tuple(df.columns)
            dfs.append(df)
            flows.append(df.flow)
        total = flows[0] + flows[1]
        where_zero = np.where(total == 0.)
        for df, f in zip(dfs, flows):
            for col in drop_zeros_from:
                if col in drop_zeros_from:
                    df.values[where_zero, drop_zeros_from.index(col)] = np.NaN
                df[col] *= f / total
        df = pd.concat(dfs, keys=labels).ffill().bfill()
        df = df.loc[labels[0]].join(df.loc[labels[1]], lsuffix='_' + labels[0], rsuffix='_' + labels[1])
        cols = list(df.columns)
        is_illicit = (df['flow_illicit'].values > 0.).astype(np.int32)
        df2 = df[cols[:len(cols) // 2]].values  + df[cols[(len(cols)) // 2:]].values
        df2 = pd.DataFrame(df2, columns=old_cols)
        df2['is_illicit'] = is_illicit
        df2.set_index(df.index, inplace=True)
        return df, df2

    def get_ml_features(self, df2=None, y=None, **kw):
        _, df2 = self.transform()
        yn = 'is_illicit'
        y = df2[[yn]]
        X = df2[[col for col in tuple(df2.columns) if col != yn]]
        y = np.atleast_2d(y.values[:, 0])
        return MLDataset({'features': xr.DataArray(X.values,
                         coords=[('space', np.array(X.index)),
                                 ('layer', np.array(list(filter(lambda x: x != yn,
                                                    df2.columns))))],
                         dims=('space', 'layer',))}), y



class Log(Step):
    use_log = False
    def transform(self, X, y=None, **kw):
        if isinstance(X, tuple) and len(X) == 2:
            X, y = X
        if self.get_params()['use_log']:
            X = X.copy()
            X[:, -1] = np.log10(X[:,-1] + 1)
        return X


def make_pipe(estimator=None):
    log_trans = Log()
    scaler = preprocessing.MinMaxScaler()
    poly = preprocessing.PolynomialFeatures()
    pca = decomposition.PCA()
    estimator = estimator or linear_model.LogisticRegression()
    names = ('flat', 'log','scaler', 'poly', 'pca', 'est')
    class Flat(Step):
        def transform(self, X, y=None, **kw):
            return X.to_features().to_xy_arrays()
    s = (Flat(), log_trans, scaler, poly, pca, estimator)
    pipe = Pipeline(list(zip(names, s)))
    return pipe


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


param_distributions = {
    'est__class_weight': ['balanced', None],
    'est__fit_intercept': [True, False],
    'est__C': [0.01, 0.1, 1, 10, 100],
    'log__use_log': [True, False],
    'poly__degree': [2, 1],
    'poly__interaction_only': [True, False],
    'poly__include_bias': [True, False],
    'pca__n_components': list(range(2, 4)),
    'pca__whiten': [True, False],
}

def density_plot(ds):
    density_grid = gridmatrix(ds, diagonal_type=hv.Distribution, chart_type=hv.Bivariate)
    point_grid = gridmatrix(ds, diagonal_type=hv.Distribution, chart_type=hv.Points)

    point_grid = point_grid.map(lambda x: hv.Overlay(), hv.Distribution)
    dens = density_grid * point_grid
    return dens

def plot(ds):
    mx = ds.data.flow.values.max()
    s = lambda c: c.opts(style={'Area': {'fill_alpha': 0.3}})
    s2 = lambda c: c.opts(style={'Curve': {'width': 800}})
    f, t, c, i, w = (s2(hv.Curve(ds.data.flow, label='Flow (GPM)')),
                     s2(hv.Curve(ds.data.temp, label='Temperature (F)')),
                     s2(hv.Curve(ds.data.cond, label='Conductivity (uS/cm)')),
                     s(hv.Area(ds.data.is_illicit * mx * 0.4, label='Has Illicit Discharge (1=Yes)')),
                     s(hv.Area(ds.data.did_warning * mx * 0.6, label='Created a warning (1=yes)')))
    return (f, t, c, i, w)


def _get_p(item, word, attr):
    return item[word + attr]


def _linspace(item, word, agg):
    conv = ('Convolution_Weight1', 'Convolution_Weight2', 'Convolution_Steps',)
    out = (_get_p(item, word, attr) for attr in conv )
    wts = np.linspace(*out)
    wts /= getattr(wts, agg)()
    return wts


def _temp_cond_params(item, word):
    wts = _linspace(item, word, 'sum')
    fl_mn = [_get_p(item, word, attr) for attr in ('Low_Flow_Mean', 'High_Flow_Mean')]
    fl_std = [_get_p(item, word, attr) for attr in ('Low_Flow_Stdev', 'High_Flow_Stdev')]
    return fl_mn, fl_std, wts


def _flow_params(item):
    return dict(prob_dry_to_wet=item['Prob_Dry_To_Wet'],
                prob_wet_to_dry=item['Prob_Wet_To_Dry'],
                flow_weights=_linspace(item, 'Flow_', 'sum'),
                flow_log_sigma=item['Log10_Sigma'],
                flow_log_mean=item['Log10_Mean'],
                is_wet_divisor=100 / item['Low_Flow_Threshold_As_Percent_Of_Mean'],
                flow_hour=np.ones(24),
                flow_month=item['flow_month'])

def _illicit(item):

    hours = np.ones(24, dtype=np.float64)
    hours[slice(*item['Peak_Hours_Of_Illicit_Discharge'])] = item['Illicit_Discharge_Peak_To_Base']
    hours /= hours.mean()
    return dict(prob_dry_to_wet=item['Prob_Dry_To_Wet'],
                prob_wet_to_dry=item['Prob_Wet_To_Dry'],
                flow_hour=hours,
                flow_month=item['flow_month'],
                flow_weights=_linspace(item, 'Flow_', 'sum'),
                flow_log_mean=item['Illicit_Flow_Log10_Mean'],
                flow_log_sigma=item['Illicit_Flow_Log10_Sigma'],
                conductivity_means=[item['Illicit_Discharge_Conductivity_Mean']] * 2,
                temperature_means=[item['Illicit_Discharge_Temperature_Mean'] ] * 2,
                conductivity_stdevs=[item['Illicit_Discharge_Conductivity_Stdev']] * 2,
                temperature_stdevs=[item['Illicit_Discharge_Temperature_Stdev']] * 2)


