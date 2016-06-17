import pickle

import numpy as np

def serialize(output_tag, model, **kwargs):
    np.savez(output_tag + '.npz', centroids=model.cluster_centers_)
    with open(output_tag + '.pkl', 'wb') as f:
        f.write(pickle.dumps(kwargs))
