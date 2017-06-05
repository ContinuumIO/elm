from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
import numpy as np

from elm.config.dask_settings import client_context
from elm.model_selection.kmeans import kmeans_model_averaging, kmeans_aic
from elm.pipeline import steps, Pipeline
from elm.readers import *
from elm.sample_util.metadata_selection import meta_is_day

ELM_EXAMPLE_DATA_PATH = os.environ['ELM_EXAMPLE_DATA_PATH']
band_specs = list(map(lambda x: BandSpec(**x),
        [{'search_key': 'long_name', 'search_value': "Band 1 ", 'name': 'band_1'},
         {'search_key': 'long_name', 'search_value': "Band 2 ", 'name': 'band_2'},
         {'search_key': 'long_name', 'search_value': "Band 3 ", 'name': 'band_3'},
         {'search_key': 'long_name', 'search_value': "Band 4 ", 'name': 'band_4'},
         {'search_key': 'long_name', 'search_value': "Band 5 ", 'name': 'band_5'},
         {'search_key': 'long_name', 'search_value': "Band 6 ", 'name': 'band_6'},
         {'search_key': 'long_name', 'search_value': "Band 7 ", 'name': 'band_7'},
         {'search_key': 'long_name', 'search_value': "Band 9 ", 'name': 'band_9'},
         {'search_key': 'long_name', 'search_value': "Band 10 ", 'name': 'band_10'},
         {'search_key': 'long_name', 'search_value': "Band 11 ", 'name': 'band_11'}]))
HDF4_FILES = [f for f in glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH, 'hdf4', '*hdf'))
              if meta_is_day(load_hdf4_meta(f))]

def sampler(fname, **kw):
    return (load_array(fname, band_specs=band_specs), None, None)

data_source = {
    'sampler': sampler,
    'args_list': HDF4_FILES,
}

pipeline_steps = [steps.Flatten(),
                  ('scaler', steps.StandardScaler()),
                  ('pca', steps.Transform(IncrementalPCA(n_components=4), partial_fit_batches=2)),
                  ('kmeans', MiniBatchKMeans(n_clusters=4, compute_labels=True)),]
pipeline = Pipeline(pipeline_steps,
                    scoring=kmeans_aic,
                    scoring_kwargs=dict(score_weights=[-1]))

def ensemble_init_func(pipe, **kw):
    return [pipe.new_with_params(kmeans__n_clusters=np.random.choice(range(6, 10)))
            for _ in range(4)]

ensemble_kwargs = {
    'model_selection': kmeans_model_averaging,
    'model_selection_kwargs': {
        'drop_n': 2,
        'evolve_n': 2,
    },
    'ensemble_init_func': ensemble_init_func,
    'ngen': 3,
    'partial_fit_batches': 2,
    'saved_ensemble_size': 4,
}

def main(pipe=None):
    with client_context() as client:
        ensemble_kwargs['client'] = client
        if pipe is None:
            pipe = pipeline
        pipe.fit_ensemble(**data_source, **ensemble_kwargs)
        pred = pipe.predict_many(**data_source, **ensemble_kwargs)
    ensemble_kwargs.pop('client')
    return pipe, pred

if __name__ == '__main__':
    pipe, pred = main()
    if 'plot' in sys.argv:
        pred[0].predict.plot.pcolormesh()
        plt.show()
