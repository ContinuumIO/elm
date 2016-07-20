from functools import partial
import pytest

import numpy as np

from elm.sample_util.geo_selection import points_in_poly, point_in_poly

def test_points_in_poly():
    poly = np.array([[1,1],[1, 0], [0, 0], [0, 1]], dtype=np.float64)
    pts = np.random.uniform(0, 1, 100).reshape(50, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    for xi, yi in zip(x, y):
        assert point_in_poly(xi, yi, poly)
    assert points_in_poly(unique_x=x, unique_y=y, include_polys=[poly]).shape[0] == x.size * y.size
    poly2 = poly + 2
    polys = [poly, poly2]
    assert points_in_poly(unique_x=x, unique_y=y, include_polys=polys).shape[0] == x.size * y.size
    pts2 = 10 + pts
    x = pts2[:, 0]
    y = pts2[:, 1]
    assert points_in_poly(unique_x=x, unique_y=y, include_polys=polys).shape[0] == 0
    assert points_in_poly(unique_x=x, unique_y=y, exclude_polys=[poly2]).shape[0] == x.size * y.size
    pts_bad = pts + 1000
    for xi, yi in zip(pts_bad[:, 0], pts_bad[:, 1]):
        assert not point_in_poly(xi, yi, poly)
    assert points_in_poly(unique_x=pts_bad[:, 0], unique_y=pts_bad[:, 1], include_polys=polys).size == 0
    assert points_in_poly(unique_x=pts_bad[:, 0], unique_y=pts_bad[:, 1], include_polys=[poly]).size == 0
    assert points_in_poly(unique_x=pts_bad[:, 0], unique_y=pts_bad[:, 1], exclude_polys=polys).shape[0] == x.size * y.size
    assert points_in_poly(unique_x=pts_bad[:, 0], unique_y=pts_bad[:, 1], exclude_polys=[poly]).shape[0] == x.size * y.size
    poly = np.array([[0, 0],[1, 1],[0, 1]], dtype=np.float64)
    assert not point_in_poly(1., 0., poly)
    assert not point_in_poly(0., -1e-6, poly)
    assert not point_in_poly(-1e-6, 0., poly)
    assert not point_in_poly(0., 1. + 1e-6, poly)
    polys = [poly, poly + 100]
    a = lambda x: np.array([x], dtype=np.float64)
    for answer, inc_exc in zip((0, 1), ('include_polys', 'exclude_polys')):
        pip = partial(points_in_poly, **{inc_exc: [poly]})
        assert pip(unique_x=a(1), unique_y=a(0)).shape[0] == answer
        assert pip(unique_x=a(0), unique_y=a(-1e-6)).shape[0] == answer
        assert pip(unique_x=a(0), unique_y=a(1 + 1e-6)).shape[0] == answer
        pip = partial(points_in_poly, **{inc_exc: polys})
        assert pip(unique_x=a(0), unique_y=a(1 + 1e-6)).shape[0] == answer
        assert pip(unique_x=a(1), unique_y=a(0)).shape[0] == answer
        assert pip(unique_x=a(0), unique_y=a(-1e-6)).shape[0] == answer

@pytest.mark.xfail
def test_filter_band_data():
    raise NotImplementedError("Need to test the filter of band data")

