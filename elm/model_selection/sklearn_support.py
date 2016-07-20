from elm.config.util import import_callable
PARTIAL_FIT_MODEL_STR = (
    'sklearn.naive_bayes:MultinomialNB',
    'sklearn.naive_bayes:BernoulliNB',
    'sklearn.linear_model:Perceptron',
    'sklearn.linear_model:SGDClassifier',
    'sklearn.linear_model:PassiveAggressiveClassifier',
    'sklearn.linear_model:SGDRegressor',
    'sklearn.linear_model:PassiveAggressiveRegressor',
    'sklearn.cluster:MiniBatchKMeans',
    'sklearn.decomposition:MiniBatchDictionaryLearning',
    'sklearn.decomposition:IncrementalPCA',
)

PARTIAL_FIT_MODEL_DICT = {k: import_callable(k)
                          for k in PARTIAL_FIT_MODEL_STR}
