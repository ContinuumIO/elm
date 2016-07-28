
import sklearn.metrics as sk_metrics

#Classification
suffixes = ('_weighted', '_samples', '_micro', '_macro')
CLASSIFIER_METRICS = {
    'accuracy_score':  sk_metrics.accuracy_score,
    'average_precision': sk_metrics.average_precision_score,
    'f1':    sk_metrics.f1_score, #    for binary targets
    'log_loss':  sk_metrics.log_loss, #    requires predict_proba support
    'precision': sk_metrics.precision_score,
    'recall': sk_metrics.recall_score,
    'roc_auc':   sk_metrics.roc_auc_score,
}
CLASSIFIER_METRICS.update({'f1' + suf: CLASSIFIER_METRICS['f1']
                           for suf in suffixes})
CLASSIFIER_METRICS.update({'precision' + suf: CLASSIFIER_METRICS['precision']
                           for suf in suffixes})
CLASSIFIER_METRICS.update({'recall' + suf: CLASSIFIER_METRICS['recall']
                           for suf in suffixes})
#Clustering
CLUSTERING_METRICS = {
   'adjusted_rand_score': sk_metrics.adjusted_rand_score,
}
#Regression
REGRESSION_METRICS = {
    'mean_absolute_error':   sk_metrics.mean_absolute_error,
    'mean_squared_error':    sk_metrics.mean_squared_error,
    'median_absolute_error': sk_metrics.median_absolute_error,
    'r2':                    sk_metrics.r2_score,
}

METRICS = {}
METRICS.update(CLASSIFIER_METRICS)
METRICS.update(CLASSIFIER_METRICS)
METRICS.update(REGRESSION_METRICS)
METRICS_STR = tuple(METRICS)
