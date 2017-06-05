from __future__ import absolute_import, division, print_function, unicode_literals

'''
----------------------

``elm.pipeline.steps``
~~~~~~~~~~~~~~~~~~~~~~~~~~

elm.pipeline.steps contains classes for use in elm.pipeline.Pipeline
steps.

Example:

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import IncrementalPCA
    from elm.sample_util.make_blobs import random_elm_store
    from elm.pipeline import Pipeline, steps

    X = random_elm_store()

    bands_to_flat = steps.Flatten()
    pca_partial_fit = steps.Transform(IncrementalPCA(n_components=3), partial_fit_batches=2)

    pipe = Pipeline([bands_to_flat,
                     pca_partial_fit,
                     MiniBatchKMeans(n_clusters=5)])

    y = pipe.fit_ensemble(X, partial_fit_batches=4).predict_many(X)
    # y is a list of ElmStores from the ensemble

'''

from elm.sample_util.preproc_scale import *
from elm.sample_util.step_mixin import StepMixin
from elm.sample_util.transform import Transform
from elm.sample_util.change_coords import *
from elm.sample_util.bands_operation import *
from elm.sample_util.ts_grid_tools import *
