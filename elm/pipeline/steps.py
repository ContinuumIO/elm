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
    from earthio.filters.make_blobs import random_elm_store
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

from earthio.filters.preproc_scale import *
from earthio.filters.step_mixin import StepMixin
from earthio.filters.transform import Transform
from earthio.filters.change_coords import *
from earthio.filters.bands_operation import *
from earthio.filters.ts_grid_tools import *
