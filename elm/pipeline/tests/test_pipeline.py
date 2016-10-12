'''
This tests Pipeline:
 * Ensures pipe.fit_transform always returns X, y, sample_weight tuple,
   where X is an ElmStore
 * Checks that each of elm.pipeline.steps, such as steps.ModifySample,
   can be used in a Pipeline
 * Checks that pipe.fit returns a fitted copy of self (a Pipeline)
 * Checks that pipe.predict can be called after .fit
'''

import numpy as np

import pytest
from sklearn.cluster import KMeans

from elm.readers import *
# Below "steps" is a module of all the
# classes which can be used for Pipeline steps
from elm.pipeline import Pipeline, steps
from elm.pipeline.tests.util import random_elm_store

data_source = {'sampler': random_elm_store}


def get_y(es, y=None, sample_weight=None, **kwargs):
    ''' Example function to get Y data

    A function that returns Y data in a Pipeline
    can return a tuple of 3 items (X, y, sample_weight)

    This function has the signature of a Pipeline
    custom function (X, y=None, sample_weight=None, **kwargs)
    '''
    if isinstance(es, ElmStore):
        # ElmStore test
        y = np.mean(es.flat.values, axis=1)
    else:
        # numpy array test
        y = np.mean(es, axis=1)
    mean_mean = np.mean(y)
    y2 = y.copy()
    y2[y > mean_mean] = 1
    y2[y < mean_mean] = 0
    return (es, y2, sample_weight)

# Example 4 step pipeline
flat_poly_var_kmeans = [('flat', steps.Flatten('C')),
                        ('poly', steps.PolynomialFeatures(interaction_only=True),),
                        ('var', steps.VarianceThreshold(threshold=0.00000001)),
                        ('kmeans', KMeans(n_clusters=2))]


def test_simple():

    p = Pipeline([('a', steps.Flatten('C'))])
    # fit_transform should always return (X, y, sample_weight)
    X, y, sample_weight = p.fit_transform(**data_source)
    assert isinstance(X, ElmStore)
    assert hasattr(X, 'flat')
    assert y is None
    assert sample_weight is None


def test_poly():
    s = flat_poly_var_kmeans
    p = Pipeline(s[:1])
    flat, y, sample_weight = p.fit_transform(**data_source)
    assert hasattr(flat, 'flat')
    p = Pipeline(s[:2])
    more_cols, _, _ = p.fit_transform(**data_source)
    assert more_cols.flat.shape[1] > flat.flat.shape[1]
    p = Pipeline(s[:3])
    feat_sel = p.fit_transform(**data_source)
    assert isinstance(feat_sel, tuple)
    p = Pipeline(s) # thru KMeans
    # fit should always return a Pipeline instance (self after fitting)
    fitted = p.fit(**data_source)
    assert isinstance(fitted, Pipeline)
    assert isinstance(fitted.steps[-1][-1], KMeans)
    assert fitted._estimator.cluster_centers_.shape[0] == fitted.get_params()['kmeans__n_clusters']
    # predict should return KMeans's predict output
    pred = p.predict(**data_source)
    # fit_transform here should return the transform of the KMeans,
    # the distances in each dimension to the cluster centers.
    out = p.fit_transform(**data_source)
    assert isinstance(out, tuple) and len(out) == 3
    X, _, _ = out
    assert X.shape[0] == pred.size


def test_set_params_get_params():
    '''Assert setting with double underscore
    parameter names will work ok'''
    p = Pipeline(flat_poly_var_kmeans)
    kw = dict(kmeans__n_clusters=9,
              poly__interaction_only=False,
              var__threshold=1e-8)
    p.set_params(**kw)
    params = p.get_params()
    for k, v in kw.items():
        assert k in params and params[k] == v
    with pytest.raises(ValueError):
        p.set_params(kmeans_n_clusters=9) # no double underscore


def test_modify_sample():
    '''steps.ModifySample should take any function and call it in Pipeline.

    The signature of the function should be:

    func(X, y=None, sample_weight=None, **kwargs)

    and it should return a tuple of:

    (X, y, sample_weight)

    '''
    p = Pipeline([steps.Flatten('C'), steps.ModifySample(get_y)])
    X, y, sample_weight = p.fit_transform(**data_source)
    assert X is not None
    assert isinstance(y, np.ndarray)


def test_predict():
    p = Pipeline(flat_poly_var_kmeans)
    # sample below is X, y, sample_weight
    sample = p.create_sample(**data_source)
    # fitted is a Pipeline instance (it returns self after fitting)
    fitted = p.fit(*sample)
    # this should be a numpy array
    pred = fitted.predict(*sample)
    assert isinstance(pred, np.ndarray)

selectors = ( 'SelectFdr',
 'SelectFpr',
 'SelectFromModel',
 'SelectFwe',
 'SelectKBest',
 'SelectPercentile',
 'VarianceThreshold')

@pytest.mark.parametrize('feat_cls', selectors)
def test_feature_selection(feat_cls):
    pytest.xfail('This test doesnt test anything yet')
    step_cls = getattr(steps, feat_cls)
    init_kwargs = {} # come up with some initialization kwargs
    p = Pipeline([steps.Flatten('C'),
                  steps.ModifySample(get_y),
                  step_cls(**init_kwargs)]) #
    X, y, sample_weight = p.fit_transform(**data_source)
    # do some assertions to make
    # sure selector was called


scalers_encoders = ('StandardScaler',
 'Binarizer',
 'FunctionTransformer',
 'GenericUnivariateSelect',
 'Imputer',
 'KernelCenterer',
 'LabelBinarizer',
 'LabelEncoder',
 'MaxAbsScaler',
 'MinMaxScaler',
 'MultiLabelBinarizer',
 'Normalizer',
 'OneHotEncoder',
 'PolynomialFeatures',)
@pytest.mark.parametrize('scale_encode_cls', scalers_encoders)
def test_sklearn_preproc(scale_encode_cls):
    pytest.xfail('This test doesnt test anything yet')
    step_cls = getattr(steps, scale_encode_cls)
    init_kwargs = {} # come up with some initialization kwargs
    p = Pipeline([steps.Flatten('C'), step_cls(**init_kwargs)]) #
    X, y, sample_weight = p.fit_transform(**data_source)
    # TODO assertions to make sure scale_encode class's fit_transform was
    # called
    # Note some may expect different shapes of X, y (change data_source)

# The following also need tests
TODO_ALSO_TEST = ('Agg',
 'DropNaRows',
 'InverseFlatten',
 'ModifySample',)
