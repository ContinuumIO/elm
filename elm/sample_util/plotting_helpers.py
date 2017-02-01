'''
------------------------------------

``elm.sample_util.plotting_helpers``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
from collections import Sequence
import matplotlib.pyplot as plt
import numpy as np

def plot_3d(X, bands, title='', scale=None, axis_labels=True,
            **imshow_kwargs):
    '''Plot a true or pseudo color image of 3 bands

    Parameters:
        :X: ElmStore or xarray.Dataset
        :bands: list of 3 band names that are in X
        :title: title for figure
        :scale: divide all values by this (e.g. 2\*\* 16 for uint16)
        :axis_labels: True / False show axis_labels
        :\*\*imshow_kwargs: passed to imshow

    Returns:
        :(arr, fig): where arr is the 3-D numpy array and fig is the figure

    '''
    arr = None
    scale = 1 if scale is None else scale
    for idx, band in enumerate(bands):
        val = getattr(X, band).values
        if idx == 0:
            arr = np.empty((val.shape) + (len(bands),), dtype=np.float32)
        if isinstance(scale, Sequence):
            s = scale[idx]
        else:
            s = scale
        arr[:, :, idx] = val.astype(np.float64) / s
    plt.imshow(arr, **imshow_kwargs)
    plt.title('{:^100}'.format(title))
    fig = plt.gcf()
    if not axis_labels:
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
    return (arr, fig)
