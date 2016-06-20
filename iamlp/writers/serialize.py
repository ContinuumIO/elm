import pickle

import numpy as np

def serialize(output_tag, model, **kwargs):

    np.savez(output_tag + '.npz', centroids=model.cluster_centers_.copy())
    filtered = {k: v for k, v in kwargs.items() if not callable(v)}
    with open(output_tag + '.pkl', 'wb') as f:
        f.write(pickle.dumps(filtered))
