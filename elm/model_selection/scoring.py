'''
----------------------------

``elm.model_selection.scroring``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This module scores models, creating a ._score attribute
that can be used for sorted members of an ensemble

'''

import copy

import sklearn.metrics as sk_metrics

from elm.config.util import  import_callable
from elm.model_selection.util import filter_kwargs_to_func
from elm.model_selection.metrics import METRICS
from elm.model_selection.util import get_args_kwargs_defaults


def import_scorer(scoring):
    '''Import a scoring function or find it in METRICS'''
    if not hasattr(scoring, 'fit'):
        if scoring in METRICS:
            scoring = import_callable(METRICS[scoring])
            requires_y = True
        else:
            scoring = import_callable(scoring)
            required_args, kwargs, has_var_kwargs = get_args_kwargs_defaults(scoring)
            requires_y = 'y_true' in required_args
    return (scoring, requires_y)


def make_scorer(scoring, **scoring_kwargs):
    func_kwargs = filter_kwargs_to_func(scoring, **scoring_kwargs)
    score_weights = scoring_kwargs.get('score_weights')
    gb = scoring_kwargs.get('greater_is_better')
    if gb is None and score_weights is not None and len(score_weights) == 1:
        sw = score_weights[0]
        scoring_kwargs['greater_is_better'] = True if sw == 1 else False
    elif gb is None:
        raise ValueError('Must provide greater_is_better in this case where score_weights is not given or longer than one element')
    scorer = sk_metrics.make_scorer(scoring,
                                 greater_is_better=scoring_kwargs.get('greater_is_better', True),
                                 needs_proba=scoring_kwargs.get('needs_proba', False),
                                 needs_threshold=scoring_kwargs.get('needs_threshold', False),
                                 **func_kwargs)
    return scorer


def _score_one_model_with_y_true(model,
                                scoring,
                                X,
                                y_true,
                                sample_weight=None,
                                **kwargs):
    if scoring is None:
        kw = copy.deepcopy(kwargs)
        kw['sample_weight'] = sample_weight
        kwargs_to_scoring = filter_kwargs_to_func(model.score, **kw)
        return model._estimator.score(X, y_true, **kwargs)
    if not isinstance(scoring, sk_metrics.scorer._PredictScorer):
        scorer = make_scorer(scoring, **kwargs)
    else:
        scorer = scoring
    # now scorer has signature:
    #__call__(self, estimator, X, y_true, sample_weight=None)
    return scorer(model, X, y_true, sample_weight=sample_weight)



def _score_one_model_no_y_true(model,
                               scoring,
                               X,
                               sample_weight=None,
                               **kwargs):
    kwargs_to_scoring = copy.deepcopy(kwargs)
    kwargs_to_scoring['sample_weight'] = sample_weight
    if scoring is None:
        kwargs = filter_kwargs_to_func(model.score, **kwargs_to_scoring)
        return model.score(X, **kwargs)
    kwargs_to_scoring = filter_kwargs_to_func(scoring,
                                            **kwargs_to_scoring)

    return scoring(model, X, **kwargs_to_scoring)



def score_one_model(model,
                    scoring,
                    X,
                    y=None,
                    sample_weight=None,
                    **kwargs):
    '''Score model with scoring function, adding ._score attribute to model

    Parameters:
        :model:   elm.pipeline.Pipeline instance
        :scoring: A scorer in sklearn.metrics or callable of the form "mypackage.mymodule:myfunc"
        :X:       elm.readers.ElmStore instance
        :y:       numpy array y data, if needed
        :sample_weight: ignored
        :kwargs:  keywords to scoring function, such as:

            * :greater_is_better: True if high scores are good
            * :needs_proba:    Estimator needs proba
            * :needs_threshold: Estimator needs threshold
            
    '''
    if scoring is None:
        if not hasattr(model, 'score') or not callable(model.score):
            raise ValueError('Cannot score model.  No scoring given and '
                             'model has no "score" callable attribute')
        requires_y = False
    else:
        scoring, requires_y = import_scorer(scoring)
    if requires_y:
        model._score = _score_one_model_with_y_true(model,
                                                    scoring,
                                                    X,
                                                    y_true=y,
                                                    sample_weight=None,
                                                    **kwargs)
    else:
        model._score = _score_one_model_no_y_true(model,
                        scoring,
                        X,
                        sample_weight=None,
                        **kwargs)
    return model
