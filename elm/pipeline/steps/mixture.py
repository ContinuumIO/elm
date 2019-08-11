'''
elm.pipeline.steps.mixture

Wraps sklearn.mixture for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.mixture import BayesianGaussianMixture as _BayesianGaussianMixture
from sklearn.mixture import GaussianMixture as _GaussianMixture



class BayesianGaussianMixture(SklearnMixin, _BayesianGaussianMixture):
    _cls = _BayesianGaussianMixture
    __init__ = _BayesianGaussianMixture.__init__



class GaussianMixture(SklearnMixin, _GaussianMixture):
    _cls = _GaussianMixture
    __init__ = _GaussianMixture.__init__

