'''
elm.pipeline.steps.naive_bayes

Wraps sklearn.naive_bayes for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.naive_bayes import BaseDiscreteNB as _BaseDiscreteNB
from sklearn.naive_bayes import BaseNB as _BaseNB
from sklearn.naive_bayes import BernoulliNB as _BernoulliNB
from sklearn.naive_bayes import GaussianNB as _GaussianNB
from sklearn.naive_bayes import MultinomialNB as _MultinomialNB



class BaseDiscreteNB(SklearnMixin, _BaseDiscreteNB):
    _cls = _BaseDiscreteNB
    __init__ = _BaseDiscreteNB.__init__



class BaseNB(SklearnMixin, _BaseNB):
    _cls = _BaseNB
    __init__ = _BaseNB.__init__



class BernoulliNB(SklearnMixin, _BernoulliNB):
    _cls = _BernoulliNB
    __init__ = _BernoulliNB.__init__



class GaussianNB(SklearnMixin, _GaussianNB):
    _cls = _GaussianNB
    __init__ = _GaussianNB.__init__



class MultinomialNB(SklearnMixin, _MultinomialNB):
    _cls = _MultinomialNB
    __init__ = _MultinomialNB.__init__

