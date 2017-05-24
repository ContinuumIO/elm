# set to debug, will plot out polygons etc
_DEBUG = False

from elm.config.tests.fixtures import *
from elm.sample_util.polygon_tools import (points_in_polys,
                                           vec_points_in_polys,
                                           plot_poly,
                                           close_poly)
import numpy as np
from itertools import product

if _DEBUG:
    import matplotlib.pyplot as plt


def test_points_in_poly_functions():
    # Tests the point in polygon functionality.

    # These polygons are "specially" designed. There are three, a quadrilateral,
    # a backwards C shape and a square.
    # The quad poly is defined clockwise and hangs off the grid in the Y direction
    # and has:
    #  * Two of the edges intersect a point somewhere on their length.
    # The backwards C poly is defined anti-clockwise and has:
    #  * A flat bottom edge for edge detection testing.
    #  * The third and sixth vertex is exactly on a grid point for testing
    #    point detection.
    #  * Hangs off the grid in the X direction
    # The combination of these polygons creates an overlapping shape which has a
    # hole in the middle (it's a lumpy doughnut). In the hole is a single
    # point (at [6, 3]) which is not to be selected.
    # The square poly is defined anti-clockwise and its edges align to the underlying
    # grid and surround a single point. This polygon does not overlap the others and
    # is present to check edge detection works on all edges and vertices.

    quad_poly_x = np.array([0.5, 5.7, 7.5, 2.5])
    quad_poly_y = np.array([0.5, 6.5, 5.5, -1.5])
    quad_poly = np.array([quad_poly_x, quad_poly_y]).T

    C_poly_x = np.array([0.5, 7.7, 7., 1.5, 1., 6, 6.5, 1.5])
    C_poly_y = np.array([1., 1., 6, 6.5, 5.5, 4., 3., 2.])
    C_poly = np.array([C_poly_x, C_poly_y]).T

    sq_poly_x = np.array([5., 7., 7., 5.])
    sq_poly_y = np.array([7., 7., 9., 9.])
    sq_poly = np.array([sq_poly_x, sq_poly_y]).T

    all_polys = (quad_poly, C_poly, sq_poly)

    # unique vectors describing the grid
    scal = 10.
    x = np.arange(0, scal)
    y = np.arange(0, scal)
    # and a grid
    (X, Y) = np.meshgrid(x, y)
    Xr = X.ravel()
    Yr = Y.ravel()

    def grid_func(polys, inon, closedPolys):
        return points_in_polys(Xr, Yr, polys, inon, closedPolys)

    def vect_func(polys, inon, closedPolys):
        return vec_points_in_polys(x, y, polys, inon, closedPolys)

    # functions to test, with their grid input wrapped
    funcs = [grid_func, vect_func]

    # There are 100 points on the grid in total.
    sz = Xr.size

    # check the functions, and try with open and closed polygons
    for f, closed in product(funcs, [False, True]):
        if closed is True:
            # the polys were defined open so close them here.
            closed_p = []
            for p in all_polys:
                vx = p[:, 0]
                vy = p[:, 1]
                (Px, Py) = close_poly(vx, vy)
                closed_p.append(np.array([Px, Py]).T)
            polys = tuple(closed_p)
        else:
            # use the defined polys
            polys = all_polys

        # With inon = True:
        inon = True
        if _DEBUG:
            inpoly = f(polys, inon, closed)
            plt.figure()
            plt.scatter(Xr[inpoly == 1], Yr[inpoly == 1], c='r', marker='x')
            plt.hold
            plt.scatter(Xr[inpoly == 0], Yr[inpoly == 0], c='y')
            for p in polys:
                plot_poly(plt, p[:, 0], p[:, 1])
            plt.show()

        # From a manual count...
        # The quadrilateral has 21 points in the polygon
        inpoly = f((polys[0],), inon, closed)
        assert(np.count_nonzero(inpoly) == 21)
        assert(inpoly.size == sz)
        # The backwards C has 27 points in the polygon
        inpoly = f((polys[1],), inon, closed)
        assert(np.count_nonzero(inpoly) == 27)
        assert(inpoly.size == sz)
        # The combined "doughnut" has 35 (removed double counting!)
        inpoly = f(polys[:2], inon, closed)
        assert(np.count_nonzero(inpoly) == 35)
        assert(inpoly.size == sz)
        # The square has 9 points in the polygon
        # This is a total of 44 in and 56 out.
        inpoly = f(polys, inon, closed)
        assert(np.count_nonzero(inpoly) == 44)
        assert(inpoly.size == sz)

        # With inon = False:
        inon = False
        if _DEBUG:
            inpoly = f(polys, inon, closed)
            plt.figure()
            plt.scatter(Xr[inpoly == 1], Yr[inpoly == 1], c='r', marker='x')
            plt.hold
            plt.scatter(Xr[inpoly == 0], Yr[inpoly == 0], c='y')
            for p in polys:
                plot_poly(plt, p[:, 0], p[:, 1])
            plt.show()

        # From a manual count...
        # The quadrilateral has 19 points in the polygon
        inpoly = f((polys[0],), inon, closed)
        assert(np.count_nonzero(inpoly) == 19)
        assert(inpoly.size == sz)
        # The backwards C has 18 points in the polygon
        inpoly = f((polys[1],), inon, closed)
        assert(np.count_nonzero(inpoly) == 18)
        assert(inpoly.size == sz)
        # The combined "doughnut" has 30 (removed double counting!)
        inpoly = f(polys[:2], inon, closed)
        assert(np.count_nonzero(inpoly) == 30)
        assert(inpoly.size == sz)
        # The square has 1 points in the polygon
        # This is a total of 31 in 69 out.
        inpoly = f(polys, inon, closed)
        assert(np.count_nonzero(inpoly) == 31)
        assert(inpoly.size == sz)
