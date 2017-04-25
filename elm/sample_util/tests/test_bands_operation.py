import numpy as np


from elm.sample_util.make_blobs import random_elm_store
from elm.sample_util.bands_operation import *

def setup():
    X = random_elm_store()
    band1, band2 = (np.random.choice(X.band_order) for _ in range(2))
    return X, band1, band2


def test_diff():
    X, band1, band2 = setup()
    spec = dict(abc=(band1, band2))
    Xnew, _, _ = BandsDiff(spec=spec).fit_transform(X)
    assert np.all(Xnew.abc.values == getattr(X, band1).values - getattr(X, band2).values)

def test_normed_diff():
    X, band1, band2 = setup()
    spec = dict(abc=(band1, band2))
    Xnew, _, _ = NormedBandsDiff(spec=spec).fit_transform(X)
    b1 = getattr(X, band1).values
    b2 = getattr(X, band2).values
    nd = (b1 - b2) / (b1 + b2)
    assert np.all(Xnew.abc.values == nd)

def test_sum():
    X, band1, band2 = setup()
    spec = dict(abc=(band1, band2))
    Xnew, _, _ = BandsSum(spec=spec).fit_transform(X)
    assert np.all(Xnew.abc.values == getattr(X, band1).values + getattr(X, band2).values)

def test_ratio():
    X, band1, band2 = setup()
    spec = dict(abc=(band1, band2))
    Xnew, _, _ = BandsRatio(spec=spec).fit_transform(X)
    assert np.all(Xnew.abc.values == getattr(X, band1).values / getattr(X, band2).values)
