from __future__ import absolute_import, division, print_function, unicode_literals

'''
---------------------------------

``elm.sample_util.polygon_tools``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
import numpy as np
from numba import njit
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Implementation inspired by pseudo-code and pros at:
# http://www.inf.usi.ch/hormann/papers/Hormann.2001.TPI.pdf
# These are essentially winding algorithms with a variety of shortcuts
# A bounding box is also employed as an initial filter.
# If a set of polygons overlap then once a point is included it is not
# rechecked for its presence in future polygons.


@njit
def close_poly(vx, vy):
    """
    Returns a closed form of coordinates of a polygon.
    (Essentially tack the first coordinate to the end of the list of
    coordinates to close the shape).

    Parameters:
        :vx: The x coordinates of the polygon (in drawing order)
        :vy: The y coordinates of the polygon (in drawing order)

    Returns:
        :tuple(X, Y): where X and Y correspond to closed versions of vx and vy respectively.
    """
    Px = np.empty(vx.size + 1)
    Px[:-1] = vx
    Px[-1] = vx[0]
    Py = np.empty(vy.size + 1)
    Py[:-1] = vy
    Py[-1] = vy[0]
    return (Px, Py)


@njit
def point_in_poly(x, y, vx, vy, inon=True, closedPoly=False):
    """
    Point in poly determines if a point is in a polygon.

    Parameters:
        :x: The x coordinate of the point
        :y: The y coordinate of the point
        :vx: The x coordinates of the polygon (in drawing order)
        :vy: The y coordinates of the polygon (in drawing order)
        :inon: If True consider points on edges and vertices as inside the polygon
        :closedPoly: If True the polygon is closed in the [vx, vy] coordinate definitions

    Returns:
        :nonzero: if the point is in the polygon
    """
    if not closedPoly:
        (Px, Py) = close_poly(vx, vy)
    else:
        Px = vx
        Py = vy

    w = 0
    vlty = Py < y
    vxmx = Px - x
    vymy = Py - y
    for idx in range(len(Px) - 1):

        if Py[idx + 1] == y:
            if Px[idx + 1] == x:
                return inon  # vertex
            else:
                if Py[idx] == y and ((Px[idx + 1] > x) == (Px[idx] < x)):
                    return inon  # edge

        if vlty[idx] != vlty[idx + 1]:
            if Px[idx] >= x:
                if Px[idx + 1] > x:
                    z = (Py[idx + 1] > Py[idx])
                    w = w + 2 * z - 1
                else:
                    d = (vxmx[idx] * vymy[idx + 1] - vxmx[idx + 1] * vymy[idx])
                    if d == 0:
                        return inon  # edge
                    z = (Py[idx + 1] > Py[idx])
                    if (d > 0) == z:
                        w = w + 2 * z - 1
            else:
                if Px[idx + 1] > x:
                    d = (vxmx[idx] * vymy[idx + 1] - vxmx[idx + 1] * vymy[idx])
                    if d == 0:
                        return inon  # edge
                    z = (Py[idx + 1] > Py[idx])
                    if (d > 0) == z:
                        w = w + 2 * z - 1
    return w


@njit
def points_in_polys(xs, ys, polys, inon=True, closedPolys=False):
    """
    Checks a set of points to determine those which are in a set of polygons

    Parameters:
        :xs: The x coordinates of the points to test (1D numpy array)
        :ys: The y coordinates of the points to test (must be the same size of xs)
        :polys: A tuple of numpy arrays size (N, 2), where the first column
                contains the x coordinates of a polygon and the second the y
        :inon: If True consider points on edges and vertices as inside the polygon
        :closedPolys: If True the polygons are closed in the polygons' coordinate definitions

    Returns:
        :vector: A vector the size of xs which has a nonzero at an index, i, corresponding to (xs[i], ys[i]) if the point is within the polygons.
    """
    n = xs.size
    inpoly = np.zeros(n, dtype=np.int16)
    for p in polys:
        vx = p[:, 0]
        vy = p[:, 1]
        # compute bounding box
        maxvx = np.max(vx)
        minvx = np.min(vx)
        maxvy = np.max(vy)
        minvy = np.min(vy)
        # close the poly
        if not closedPolys:
            (Px, Py) = close_poly(vx, vy)
        else:
            Px = vx
            Py = vy
        for k in range(n):
            # see if it is already in a polygon, if so, skip
            if inpoly[k] == False:
                x = xs[k]
                y = ys[k]
                if x >= minvx and x <= maxvx:
                    if y >= minvy and y <= maxvy:
                        if point_in_poly(x, y, Px, Py, inon, True):
                            inpoly[k] = True
    return inpoly


@njit
def vec_points_in_polys(x_vec, y_vec, polys, inon=True, closedPolys=False):
    """
    Checks a set of points defined by the meshgrid of two input vectors to determine
    those which are in a set of polygons.

    Parameters:
        :x_vec: The x coordinates of the points to test (1D numpy array)
        :y_vec: The y coordinates of the points to test (1D numpy array)
        :polys: A tuple of numpy arrays size (N, 2), where the first column contains the x coordinates of a polygon and the second the y.
        :inon: If True consider points on edges and vertices as inside the polygon
        :closedPolys: If True the polygons are closed in the polygons' coordinate definitions.

    Returns:
        :array: an array size (x_vec.size, y_vect.size) which has a nonzero at an index  (i, j), corresponding to:

            (X, Y) = meshgrid(x_vec, y_vec);
            (X[i], Y[j]) if the point is within the polygons.
            
    """
    nx = x_vec.size
    ny = y_vec.size
    inpoly = np.zeros((nx, ny), dtype=np.int16)
    for p in polys:
        vx = p[:, 0]
        vy = p[:, 1]
        # compute bounding box
        maxvx = np.max(vx)
        minvx = np.min(vx)
        maxvy = np.max(vy)
        minvy = np.min(vy)
        # close the poly
        if not closedPolys:
            (Px, Py) = close_poly(vx, vy)
        else:
            Px = vx
            Py = vy
        for ky in range(len(y_vec)):
            y = y_vec[ky]
            for kx in range(len(x_vec)):
                x = x_vec[kx]
                # see if it is already in a polygon, if so, skip
                if inpoly[ky, kx] == False:
                    if x >= minvx and x <= maxvx:
                        if y >= minvy and y <= maxvy:
                            if point_in_poly(x, y, Px, Py, inon, True):
                                inpoly[ky, kx] = True
    return inpoly


def plot_poly(plt, poly_x, poly_y):
    # Debug helper, will plot a polygon into plt's gcf
    length = len(poly_x) + 1
    pthcode = [Path.LINETO] * length
    pthcode[0] = Path.MOVETO
    pthcode[-1] = Path.CLOSEPOLY
    q = np.zeros([length, 2])
    q[:-1, 0] = poly_x
    q[:-1, 1] = poly_y
    q[-1, 0] = poly_x[0]
    q[-1, 1] = poly_y[0]
    path = Path(q, pthcode)
    patch = patches.PathPatch(path, facecolor='green', lw=2, alpha=0.3)
    plt.gca().add_patch(patch)
