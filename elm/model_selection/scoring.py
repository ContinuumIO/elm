import copy

import sklearn.metrics as sk_metrics

from elm.config.util import  import_callable
from elm.model_selection.util import filter_kwargs_to_func
from elm.model_selection.metrics import METRICS
def make_scorer(scoring, **scoring_kwargs):
    if not hasattr(scoring, 'fit'):
        if scoring in METRICS:
            scoring = import_callable(METRICS[scoring])
        else:
            scoring = import_callable(scoring)
    func_kwargs = filter_kwargs_to_func(scoring, **scoring_kwargs)
    scorer = sk_metrics.make_scorer(scoring,
                                 greater_is_better=scoring_kwargs.get('greater_is_better', True),
                                 needs_proba=scoring_kwargs.get('needs_proba', False),
                                 needs_threshold=scoring_kwargs.get('needs_threshold', False),
                                 **func_kwargs)
    return scorer


def _score_one_model_with_y_true(model,
                                scoring,
                                x,
                                y_true,
                                sample_weight=None,
                                **kwargs):
    if not isinstance(scoring, sk_metrics.scorer._PredictScorer):
        scorer = make_scorer(scoring, **kwargs)
    else:
        scorer = scoring
    # now scorer has signature:
    #__call__(self, estimator, x, y_true, sample_weight=None)
    return scorer(model, x, y_true, sample_weight=sample_weight)



def _score_one_model_no_y_true(model,
                               scoring,
                               x,
                               sample_weight=None,
                               **kwargs):
    kwargs_to_scoring = copy.deepcopy(kwargs)
    kwargs_to_scoring['sample_weight'] = sample_weight
    kwargs_to_scoring = filter_kwargs_to_func(scoring,
                                            **kwargs_to_scoring)
    return scoring(model, x, **kwargs_to_scoring)



def score_one_model(model,
                    scoring,
                    x,
                    y=None,
                    sample_weight=None,
                    **kwargs):
    if y is not None:
        model._score = _score_one_model_with_y_true(model,
                                                    scoring,
                                                    x,
                                                    y_true=y,
                                                    sample_weight=None,
                                                    **kwargs)
    else:
        scoring = import_callable(scoring)
        model._score = _score_one_model_no_y_true(model,
                        scoring,
                        x,
                        sample_weight=None,
                        **kwargs)
    return model
