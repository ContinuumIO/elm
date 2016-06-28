import copy

from iamlp.model_selection.util import get_args_kwargs_defaults
from iamlp.samplers import Sample

PARTIAL_FIT_MODELS = [
    # Classification
    'sklearn.naive_bayes:MultinomialNB',
    'sklearn.naive_bayes:BernoulliNB',
    'sklearn.linear_model:Perceptron',
    'sklearn.linear_model:SGDClassifier',
    'sklearn.linear_model:PassiveAggressiveClassifier',
    # Regression
    'sklearn.linear_model:SGDRegressor',
    'sklearn.linear_model:PassiveAggressiveRegressor',
    # Clustering
    'sklearn.cluster:MiniBatchKMeans',
    # Decomposition
    'sklearn.decomposition:MiniBatchDictionaryLearning',
    'sklearn.decomposition:IncrementalPCA',
    'sklearn.cluster:MiniBatchKMeans',
]


def final_on_sample_step(fitter,
                         model, sample,
                         iter_offset,
                         fit_kwargs,
                         get_y_func=None,
                         get_y_kwargs=None,
                         get_weight_func=None,
                         get_weight_kwargs=None,
                         classes=None,
                      ):
    '''This is the final function called on a sample_pipeline
    or a simple sample that is input to training.  It ensures
    that:
       * Corresponding Y data are looked up for the X sample
       * The correct fit kwargs are passed to fit or partial_fit,
         depending on the method
    Params:
       fitter:  a model attribute like "fit" or "partial_fit"
       model:   a sklearn model like MiniBatchKmeans()
       sample:  a dataframe sample
       fit_kwargs: kwargs to fit_func from config
       get_y_func: a function which takes an X sample DataFrame
                   and returns the corresponding Y
       get_y_kwargs: get_y_kwargs are kwargs to get_y_func from config
       get_weight_func: a function which returns sample weights for
                        an X sample
       get_weight_kwargs: keyword args needed by get_weight_func
       classes:  if using classification, all possible classes as iterable
                 or array of integers
       '''
    args, kwargs = get_args_kwargs_defaults(fitter)
    fit_kwargs = fit_kwargs or {}
    fit_kwargs = copy.deepcopy(fit_kwargs)
    if classes is not None:
        fit_kwargs['classes'] = classes
    if 'iter_offset' in kwargs:
        fit_kwargs['iter_offset'] = iter_offset
    if 'check_input' in kwargs:
        fit_kwargs['check_input'] = True
    if 'sample_weight' in kwargs and get_weight_func is not None:
        get_weight_kwargs = get_weight_kwargs or {}
        fit_kwargs['sample_weight'] = get_weight_func(sample, **get_weight_kwargs)
    if isinstance(sample, Sample):
        sample = sample.df
    if any(a.lower() == 'y' for a in args):
        y = get_y_func(sample)
        fit_args = (sample.values, y)
    else:
        fit_args = (sample.values, )
    return fit_args, fit_kwargs
