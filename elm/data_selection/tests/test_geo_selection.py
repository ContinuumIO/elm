import pytest

import numpy as np

from elm.data_selection.geo_selection import points_in_poly, point_in_poly

def test_points_in_poly():
    poly = np.array([[1,1],[1, 0], [0, 0], [0, 1]], dtype=np.float64)
    pts = np.random.uniform(0, 1, 100).reshape(50, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    for xi, yi in zip(x, y):
        assert point_in_poly(xi, yi, poly)
    poly2 = poly + 10
    polys = np.concatenate((poly, np.ones((1, poly.shape[1]), dtype=np.float64) * np.NaN, poly2))
    poly2 = poly2.astype(np.float64)
    for xi, yi in zip(x, y):
        assert point_in_poly(xi, yi, polys)
    pts2 = 10 + pts
    x = pts2[:, 0]
    y = pts2[:, 1]
    for xi, yi in zip(x, y):
        assert point_in_poly(xi, yi, polys)
    pts_bad = pts + 1000
    for xi, yi in zip(pts_bad[:, 0], pts_bad[:, 1]):
        assert not point_in_poly(xi, yi, poly)


@pytest.mark.xfail
def test_filter_band_data():
    raise NotImplementedError("Need to test the filter of band data")