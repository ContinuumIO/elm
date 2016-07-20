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
    'sklearn.decomposition:MiniBatchDictionaryLearning',
    'sklearn.decomposition:IncrementalPCA',
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
              'RANSACRegressor',
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
                            'sklearn.decomposition:IncrementalPCA',
                            'sklearn.linear_model:LogisticRegression',
                            'sklearn.linear_model:LogisticRegressionCV',
                            'sklearn.linear_model:Perceptron',
                            'sklearn.linear_model:RandomizedLasso',
                            'sklearn.linear_model:RandomizedLogisticRegression',
                            )
ALL_MODELS_STR = LINEAR_MODELS_WITH_PREDICT_STR + FIT_TRANSFORM_MODELS_STR + \
                 PARTIAL_FIT_MODEL_STR
ALL_MODELS_DICT = {k: import_callable(k) for k in ALL_MODELS_STR}
UNSUPERVISED_MODEL_STR = [k for k, v in ALL_MODELS_DICT.items()
                          if hasattr(v, 'fit')
                          and 'y' in get_args_kwargs_defaults(v.fit)[1]]