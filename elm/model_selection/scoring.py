from __future__ import absolute_import, division, print_function, unicode_literals

'''
--------------------------------

``elm.model_selection.scoring``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This module scores models, creating a ._score attribute
that can be used for sorted members of an ensemble

'''

import copy

import sklearn.metrics as sk_metrics

from elm.config.util import  import_callable
from elm.config.func_signatures import (filter_kwargs_to_func,
                                        get_args_kwargs_defaults)
from elm.model_selection.metrics import METRICS


def import_scorer(scoring):
    '''Import a scoring function or find it in METRICS'''
    requires_y = False
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
    '''
    # TODO - this function needs refactor + docs explanation
    # It is brittle and difficult to use.
     1) Unclear how to pass in a custom scorer:
        a) Do I need to call make_scorer() myself on that function
        b) If I need to call make_scorer(), do I use the make_scorer
           above or the make_scorer from sklearn?
     2) Should a scoring function expect the model to be passed into
        it, and if so, do should I expect always a Pipeline instance
        from the elm stack or the Pipeline's _estimator attribute
        that is typically an sklearn model like LinearRegression()
     3) Should my scoring function, whether custom or standard like r2_score,
        expect that X and y are xarray data structures or numpy ones
        that have been already prepared for use with the final
        sklearn estimator
     4)
    '''
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
    # TODO should we pass model or model._estimator?
    return scorer(model._estimator, X, y_true, sample_weight=sample_weight)



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
        :X:       earthio.ElmStore instance
        :y:       numpy array y data, if needed
        :sample_weight: ignored
        :kwargs:  keywords to scoring function, such as:

            * :greater_is_better: True if high scores are good
            * :needs_proba:    Estimator needs proba
            * :needs_threshold: Estimator needs threshold

    '''
    y = y if y is not None else kwargs.get('y_true') # TODO this needs diambiguation
    model_to_pass = model
    if scoring:
        scoring, requires_y = import_scorer(scoring)
    else:
        scoring = model._estimator.score
        model_to_pass = model._estimator
    requires_y = True if y is not None else requires_y
    if requires_y:
        model._score = _score_one_model_with_y_true(model_to_pass,
                                                    scoring,
                                                    X,
                                                    y_true=y,
                                                    sample_weight=sample_weight,
                                                    **kwargs)
    else:
        model._score = _score_one_model_no_y_true(model_to_pass,
                        scoring,
                        X,
                        sample_weight=None,
                        **kwargs)
    return model
