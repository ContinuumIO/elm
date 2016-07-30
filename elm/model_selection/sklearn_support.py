from elm.config.util import import_callable
from elm.model_selection.util import get_args_kwargs_defaults
PARTIAL_FIT_MODEL_STR = (
    'sklearn.naive_bayes:MultinomialNB',
    'sklearn.naive_bayes:BernoulliNB',
    'sklearn.linear_model:Perceptron',
    'sklearn.linear_model:SGDClassifier',
    'sklearn.linear_model:PassiveAggressiveClassifier',
    'sklearn.linear_model:SGDRegressor',
    'sklearn.linear_model:PassiveAggressiveRegressor',
    'sklearn.cluster:MiniBatchKMeans',
    'sklearn.cluster:Birch',
    'sklearn.neural_network:BernoulliRBM'
)

PARTIAL_FIT_MODEL_DICT = {k: import_callable(k)
                          for k in PARTIAL_FIT_MODEL_STR}

FIT_PREDICT_MODELS_STR = ['sklearn.tree:DecisionTreeClassifier',
                      'sklearn.tree:DecisionTreeRegressor',
                      'sklearn.tree:ExtraTreeClassifier',
                      'sklearn.tree:ExtraTreeRegressor',]
linear_models_predict = ['ARDRegression',
              'BayesianRidge',
              'ElasticNet',
              'ElasticNetCV',
              'Lars',
              'LarsCV',
              'Lasso',
              'LassoCV',
              'LassoLars',
              'LassoLarsCV',
              'LassoLarsIC',
              'LinearRegression',
              'LogisticRegression',
              'LogisticRegressionCV',
              'MultiTaskElasticNet',
              'MultiTaskLassoCV',
              'MultiTaskElasticNetCV',
              'MultiTaskLasso',
              'OrthogonalMatchingPursuit',
              'OrthogonalMatchingPursuitCV',
              'PassiveAggressiveClassifier',
              'PassiveAggressiveRegressor',
              'Perceptron',
              'RidgeClassifier',
              'Ridge',
              'RidgeClassifierCV',
              'RidgeCV',
              'SGDClassifier',
              'SGDRegressor',
              'TheilSenRegressor']

LINEAR_MODELS_WITH_PREDICT_STR = ['sklearn.linear_model:{}'.format(func)
                                   for func in linear_models_predict]
LINEAR_MODELS_WITH_PREDICT_STR += ['sklearn.gaussian_process:GaussianProcess',
                                   'sklearn.discriminant_analysis:LinearDiscriminantAnalysis',
                                   'sklearn.discriminant_analysis:QuadraticDiscriminantAnalysis']
LINEAR_MODELS_WITH_PREDICT_STR = tuple(LINEAR_MODELS_WITH_PREDICT_STR)
FIT_TRANSFORM_MODELS_STR = ('sklearn.cluster:SpectralClustering',
                            'sklearn.manifold:SpectralEmbedding',
                            'sklearn.manifold:LocallyLinearEmbedding',
                            'sklearn.linear_model:LogisticRegression',
                            'sklearn.linear_model:LogisticRegressionCV',
                            'sklearn.linear_model:Perceptron',
                            'sklearn.linear_model:RandomizedLasso',
                            'sklearn.linear_model:RandomizedLogisticRegression',
                            )
MODELS_WITH_PREDICT_STR = LINEAR_MODELS_WITH_PREDICT_STR + FIT_TRANSFORM_MODELS_STR + \
                 PARTIAL_FIT_MODEL_STR
MODELS_WITH_PREDICT_DICT = {k: import_callable(k) for k in MODELS_WITH_PREDICT_STR}

#
DECOMP_PARTIAL_FIT_MODEL_STR = (
    'sklearn.decomposition:MiniBatchDictionaryLearning',
    'sklearn.decomposition:IncrementalPCA',
)
DECOMP_MODEL_STR = (
    'sklearn.decomposition:PCA',
    'sklearn.decomposition:ProjectedGradientNMF',
    'sklearn.decomposition:RandomizedPCA',
    'sklearn.decomposition:KernelPCA',
    'sklearn.decomposition:FactorAnalysis',
    'sklearn.decomposition:FastICA',
    'sklearn.decomposition:TruncatedSVD',
    'sklearn.decomposition:NMF',
    'sklearn.decomposition:SparsePCA',
    'sklearn.decomposition:MiniBatchSparsePCA',
    'sklearn.decomposition:SparseCoder',
    'sklearn.decomposition:DictionaryLearning',
    'sklearn.decomposition:LatentDirichletAllocation',
) + DECOMP_PARTIAL_FIT_MODEL_STR
UNSUPERVISED_MODEL_STR = [k for k, v in MODELS_WITH_PREDICT_DICT.items()
                          if hasattr(v, 'fit')
                          and 'y' in get_args_kwargs_defaults(v.fit)[1]]


MODELS_WITH_PREDICT_ESTIMATOR_TYPES = {k: getattr(v, '_estimator_type', None)
                              for k,v in MODELS_WITH_PREDICT_DICT.items()}




