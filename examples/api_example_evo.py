from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np

from elm.config.dask_settings import client_context
from elm.model_selection.evolve import ea_setup
from elm.model_selection.kmeans import kmeans_model_averaging, kmeans_aic
from elm.pipeline import Pipeline, steps
from elm.readers import *


from api_example import data_source

ELM_EXAMPLE_DATA_PATH = os.environ['ELM_EXAMPLE_DATA_PATH']


def make_example_y_data(X, y=None, sample_weight=None, **kwargs):
    fitted = MiniBatchKMeans(n_clusters=5).fit(X.flat.values)
    y = fitted.predict(X.flat.values)
    return (X, y, sample_weight)

pipeline_steps = [steps.Flatten(),
                  steps.ModifySample(make_example_y_data),
                  ('top_n', steps.SelectPercentile(percentile=80,score_func=f_classif)),
                  ('kmeans', MiniBatchKMeans(n_clusters=4))]
pipeline = Pipeline(pipeline_steps, scoring=kmeans_aic)
param_grid =  {
    'kmeans__n_clusters': list(range(5, 10)),
    'control': {
        'select_method': 'selNSGA2',
        'crossover_method': 'cxTwoPoint',
        'mutate_method': 'mutUniformInt',
        'init_pop': 'random',
        'indpb': 0.5,
        'mutpb': 0.9,
        'cxpb':  0.3,
        'eta':   20,
        'ngen':  2,
        'mu':    4,
        'k':     4,
        'early_stop': {'abs_change': [10], 'agg': 'all'},
        # alternatively early_stop: {percent_change: [10], agg: all}
        # alternatively early_stop: {threshold: [10], agg: any}
    }
}

evo_params = ea_setup(param_grid=param_grid,
                      param_grid_name='param_grid_example',
                      score_weights=[-1]) # minimization

def main():
    with client_context() as client:
        fitted = pipeline.fit_ea(evo_params=evo_params,
                                 client=client,
                                 saved_ensemble_size=param_grid['control']['mu'],
                                 **data_source)
        preds = pipeline.predict_many(client=client, **data_source)
    return fitted, preds


if __name__ == '__main__':
    fitted, preds = main()
    if 'plot' in sys.argv:
        preds[0].predict.plot.pcolormesh()
        plt.show()

