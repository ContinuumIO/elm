'''
elm.pipeline.steps.preprocessing

Wraps sklearn.preprocessing for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.preprocessing import Binarizer as _Binarizer
from sklearn.preprocessing import FunctionTransformer as _FunctionTransformer
from sklearn.preprocessing import Imputer as _Imputer
from sklearn.preprocessing import KernelCenterer as _KernelCenterer
from sklearn.preprocessing import LabelBinarizer as _LabelBinarizer
from sklearn.preprocessing import LabelEncoder as _LabelEncoder
from sklearn.preprocessing import MaxAbsScaler as _MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer as _MultiLabelBinarizer
from sklearn.preprocessing import Normalizer as _Normalizer
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures as _PolynomialFeatures
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer
from sklearn.preprocessing import RobustScaler as _RobustScaler
from sklearn.preprocessing import StandardScaler as _StandardScaler



class Binarizer(SklearnMixin, _Binarizer):
    _cls = _Binarizer
    __init__ = _Binarizer.__init__



class FunctionTransformer(SklearnMixin, _FunctionTransformer):
    _cls = _FunctionTransformer
    __init__ = _FunctionTransformer.__init__



class Imputer(SklearnMixin, _Imputer):
    _cls = _Imputer
    __init__ = _Imputer.__init__



class KernelCenterer(SklearnMixin, _KernelCenterer):
    _cls = _KernelCenterer
    __init__ = _KernelCenterer.__init__



class LabelBinarizer(SklearnMixin, _LabelBinarizer):
    _cls = _LabelBinarizer
    __init__ = _LabelBinarizer.__init__



class LabelEncoder(SklearnMixin, _LabelEncoder):
    _cls = _LabelEncoder
    __init__ = _LabelEncoder.__init__



class MaxAbsScaler(SklearnMixin, _MaxAbsScaler):
    _cls = _MaxAbsScaler
    __init__ = _MaxAbsScaler.__init__



class MinMaxScaler(SklearnMixin, _MinMaxScaler):
    _cls = _MinMaxScaler
    __init__ = _MinMaxScaler.__init__



class MultiLabelBinarizer(SklearnMixin, _MultiLabelBinarizer):
    _cls = _MultiLabelBinarizer
    __init__ = _MultiLabelBinarizer.__init__



class Normalizer(SklearnMixin, _Normalizer):
    _cls = _Normalizer
    __init__ = _Normalizer.__init__



class OneHotEncoder(SklearnMixin, _OneHotEncoder):
    _cls = _OneHotEncoder
    __init__ = _OneHotEncoder.__init__



class PolynomialFeatures(SklearnMixin, _PolynomialFeatures):
    _cls = _PolynomialFeatures
    __init__ = _PolynomialFeatures.__init__



class QuantileTransformer(SklearnMixin, _QuantileTransformer):
    _cls = _QuantileTransformer
    __init__ = _QuantileTransformer.__init__



class RobustScaler(SklearnMixin, _RobustScaler):
    _cls = _RobustScaler
    __init__ = _RobustScaler.__init__



class StandardScaler(SklearnMixin, _StandardScaler):
    _cls = _StandardScaler
    __init__ = _StandardScaler.__init__

