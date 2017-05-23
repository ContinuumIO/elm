ElmStore
==============================

``ElmStore``, from ``earthio``, is a fundamental data structure in ``elm`` and is the data structure used to pass arrays and metadata through each of the steps in an :doc:`Pipeline <pipeline>` (series of transformations).  An ``ElmStore`` is oriented around multi-band rasters and cubes stored in HDF4 / 5, NetCDF, or GeoTiff formats. ``ElmStore`` is a light wrapper around ``xarray.Dataset``.

This page discusses:

 * :ref:`elm-store-from-file`
 * :ref:`elm-store-constructor`
 * :ref:`elm-store-attributes`
 * :ref:`common-elm-store-transformations`

See also :doc:`API docs<api>`.

.. _elm-store-from-file:

Creating an ``ElmStore`` from File
----------------------------------
An ``ElmStore`` can be created from ``HDF4`` / ``HDF5`` or ``NetCDF`` file with ``load_array`` from ``earthio``.  The simple case is to load all bands or subdatasets from an HDF or NetCDF file:

.. code-block:: python

    from earthio import load_array
    filename = '3B-HHR-E.MS.MRG.3IMERG.20160708-S153000-E155959.0930.V03E.HDF5.nc'
    es = load_array(filename)

For GeoTiffs the argument is a directory name rather than a file name and each band is formed from individual GeoTiff files in the directory.  The following is an example with LANDSAT GeoTiffs for bands 1 through 7:

.. code-block:: python

    In [1]: from earthio import BandSpec, load_array

    In [2]: ls
    LC80150332013207LGN00_B1.TIF  LC80150332013207LGN00_B5.TIF
    LC80150332013207LGN00_B2.TIF  LC80150332013207LGN00_B6.TIF
    LC80150332013207LGN00_B3.TIF  LC80150332013207LGN00_B7.TIF
    LC80150332013207LGN00_B4.TIF  logfile.txt

    In [3]: es = load_array('.')
    In [4]: es.data_vars
    Out[4]:
    Data variables:
        band_0   (y, x) uint16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
        band_1   (y, x) uint16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
        band_2   (y, x) uint16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
        band_3   (y, x) uint16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
        band_4   (y, x) uint16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
        band_5   (y, x) uint16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
        band_6   (y, x


The example above for GeoTiffs loaded the correct bands, but labeled them in a way that may be confusing downstream in the analysis.  The following section shows how to control which bands are loaded and what they are named.

Controlling Which Bands Are Loaded
----------------------------------

Use the ``band_specs`` keyword to ``load_array`` to

 * Control which subdatasets, or bands typically, are loaded into the ``ElmStore`` and/or
 * To standardize the names of the bands (``DataArrays``) in the ``ElmStore``.

 The ``band_specs`` work slightly differently for each file type:

  * HDF4 / HDF5: The ``band_specs`` determine matching against one of the HDF4 file's subdatasets (see also `GDAL subdatasets`_).
  * NetCDF: The ``band_specs`` determine matching against one of the NetCDF file's ``variables`` metadata (`NetCDF4 python variables interface`_).
  * GeoTiff: When calling ``load_array`` for GeoTiffs, the argument is a directory (of GeoTiff files) not a single GeoTiff file.  The ``band_specs`` for a GeoTiff file determine matching based on the gdal metadata for each GeoTiff in the directory.  GeoTiffs are read using ``rasterio`` , `a wrapper around GDAl`_.

.. _GDAL subdatasets: http://www.gdal.org/gdalinfo.html
.. _NetCDF4 python variables interface: http://unidata.github.io/netcdf4-python/
.. _a wrapper around GDAl: https://mapbox.github.io/rasterio/

In simple cases ``band_specs`` can be a list of strings to match a ``NetCDF`` variable name, subdataset name, or GeoTiff file name, as shown below:

.. code-block:: python

    In [4]: from earthio import load_array
    In [5]: filename = '3B-HHR-E.MS.MRG.3IMERG.20160708-S153000-E155959.0930.V03E.HDF5.nc'
    In [6]: es = load_array(filename, band_specs=['HQobservationTime'])
    In [7]: es.data_vars
    Out[7]:
    Data variables:
        HQobservationTime  (lon, lat) timedelta64[ns] NaT NaT NaT NaT NaT NaT ...

With GeoTiffs, giving a list of strings as ``band_specs`` finds matching GeoTiff files (bands) by testing if each string is ``in`` a GeoTiff file name of the directory.  Here is an example:

.. code-block:: python

    from earthio import load_array
    dir_of_tifs = '.'
    load_array(dir_of_tifs, band_specs=["B1.TIF", "B2.TIF","B3.TIF"])


``band_specs`` can be given as a list of ``earthio.BandSpec`` objects.  The following shows an example of loading 4 bands from an ``HDF4`` file where the band name, such as ``"Band 1 "`` is found in the ``long_name`` key/value of the subdataset (band) metadata and the band names are standardized to lower case with no spaces.

.. code-block:: python

    In [1]: from earthio import BandSpec, load_array

    In [2]: band_specs = list(map(lambda x: BandSpec(**x),
       [{'search_key': 'long_name', 'search_value': "Band 1 ", 'name': 'band_1'},
       {'search_key': 'long_name', 'search_value': "Band 2 ", 'name': 'band_2'},
       {'search_key': 'long_name', 'search_value': "Band 3 ", 'name': 'band_3'},
       {'search_key': 'long_name', 'search_value': "Band 4 ", 'name': 'band_4'}]))

    In [3]: filename = 'NPP_DSRF1KD_L2GD.A2015017.h09v05.C1_03001.2015018132754.hdf'

    In [4]: es = load_array(filename, band_specs=band_specs)

    In [5]: es.data_vars
    Out[5]:
    Data variables:
        band_1   (y, x) uint16 877 877 767 659 920 935 935 918 957 989 989 789 ...
        band_2   (y, x) uint16 899 899 770 659 954 973 973 935 994 1004 1004 841 ...
        band_3   (y, x) uint16 1023 1023 880 781 1115 1141 1141 1082 1155 1154 ...
        band_4   (y, x) uint16 1258 1258 1100 1009 1374 1423 1423 1341 1408 1405 ...

Note the ``BandSpec`` objects could have also used the keyword arguments ``key_re_flags`` and ``value_re_flags`` with a list of flags passed to `re` for regular expression matching.

.. _bandspec-file-reading-control:

``BandSpec`` - File Reading Control
-----------------------------------

Here are a few more things a ``BandSpec`` can do:

 * A ``BandSpec`` can control the resolution at which a file is read (and improve loading speed).  To control resolution when loading rasters, provide ``buf_xsize`` and ``buf_ysize`` keyword arguments (integers) to ``BandSpec``.
 * A ``BandSpec`` can provide a ``window`` that subsets the file.  See `this rasterio demo`_ that shows how ``window`` is effectively interpreted in ``load_array``.
 * A ``BandSpec`` with a ``meta_to_geotransform`` callable attribute can be used to construct a ``geo_transform`` array from band metadata (e.g. when GDAL fails to detect the ``geo_transform`` accurately)
 * A ``BandSpec`` can control whether a raster is loaded with `("y", "x")`  pixel order (the default behavior that suits most top-left-corner based rasters) or `("x", "y")` pixel order.

See also the definition of ``BandSpec`` in ``earthio`` showing all the recognized fields (`snippet taken from earthio.util`_).

.. _this rasterio demo: https://sgillies.net//2013/12/21/rasterio-windows-and-masks.html

.. _snippet taken from earthio.util: https://github.com/ContinuumIO/elm/blob/master/earthio/util.py

.. code-block:: python

    @attr.s
    class BandSpec(object):
        search_key = attr.ib()
        search_value = attr.ib()
        name = attr.ib()
        key_re_flags = attr.ib(default=None)
        value_re_flags = attr.ib(default=None)
        buf_xsize = attr.ib(default=None)
        buf_ysize = attr.ib(default=None)
        window = attr.ib(default=None)
        meta_to_geotransform = attr.ib(default=None)
        stored_coords_order = attr.ib(default=('y', 'x'))

.. _elm-store-constructor:

Creating an ``ElmStore`` - Contructor
-------------------------------------
Here is an example of creating an ``ElmStore`` from ``numpy`` arrays and ``xarray.DataArrays``.  In most ways, an ``ElmStore`` is interchangeable with an ``xarray.Dataset`` (see also `docs on working with a Dataset`_.

.. _docs on working with a Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html

.. code-block:: python

    from collections import OrderedDict
    import numpy as np
    import xarray as xr
    from earthio import ElmStore

    rand_array = lambda: np.random.normal(0, 1, 1000000).reshape(-1,10)

    def sampler(**kwargs):
        bands = ['b1', 'b2', 'b3', 'b4']
        es_data = OrderedDict()
        for band in bands:
            arr = rand_array()
            y = np.arange(arr.shape[0])
            x = np.arange(arr.shape[1])
            es_data[band] = xr.DataArray(arr, coords=[('y', y), ('x', x)], dims=('y', 'x'), attrs={})
        return ElmStore(es_data, add_canvas=False)

Calling ``sampler`` above gives:

.. code-block:: python

    <elm.ElmStore>
    Dimensions:  (x: 10, y: 100000)
    Coordinates:
      * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
      * x        (x) int64 0 1 2 3 4 5 6 7 8 9
    Data variables:
        b1       (y, x) float64 1.772 -0.414 1.37 2.107 -1.306 0.9612 -0.0696 ...
        b2       (y, x) float64 0.07442 1.908 0.5816 0.06838 -2.712 0.4544 ...
        b3       (y, x) float64 -2.597 -1.893 0.05608 -0.5394 1.406 -0.6185 ...
        b4       (y, x) float64 1.054 -1.522 -0.03202 -0.02127 0.02914 -0.6757 ...
    Attributes:
        _dummy_canvas: True
        band_order: ['b1', 'b2', 'b3', 'b4']

``ElmStore`` has the initialization keyword argument ``add_canvas`` that differs from ``xarray.Dataset``.  If ``add_canvas`` is True (default), it expected that the band metadata in the ``DataArrays`` each contain a ``geo_transform`` key with a value that is a sequence of length 6.  See `the GDAL data model for more information on geo transforms`_.  In the example above each ``DataArray`` did not have a ``geo_transform`` in ``attrs`` so ``add_canvas`` was set to ``False``.  The limitation of not having a ``canvas`` attribute is inability to use some spatial reindexing transformations (e.g. ``elm.pipeline.steps.SelectCanvas`` described further below)

.. _the GDAL data model for more information on geo transforms: http://www.gdal.org/classGDALDataset.html#a5101119705f5fa2bc1344ab26f66fd1d

.. _elm-store-attributes:

Attributes of an ``ElmStore``
-----------------------------

If an ``ElmStore`` was initialized with ``add_canvas`` (the behavior in ``load_array``), then it is expected each band, or ``DataArray``, will have a ``geo_transform`` in its metadata.  The ``geo_transform`` information, in combination with the array dimensions and shape, create the ``ElmStore``'s ``canvas`` attribute.

.. code-block:: python

    In [4]: es.canvas

    Out[5]: Canvas(geo_transform=(-180.0, 0.1, 0, -90.0, 0, 0.1), buf_xsize=3600, buf_ysize=1800, dims=('lon', 'lat'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=-90.0, right=179.90000000000003, top=89.9))

The ``canvas`` is used in the ``Pipeline`` for transformations like ``elm.pipeline.steps.SelectCanvas`` which can be used to reindex all bands onto coordinates of one of the band's in the ``ElmStore``.

An ``ElmStore`` has a ``data_vars`` attribute (inherited from ``xarray.Dataset`` - `described here`_), and also has an attribute ``band_order``.  When ``elm.pipeline.steps.Flatten`` flattens the separate bands of an ``ElmStore``, ``band_order`` becomes the order of the bands in the single flattened 2-D array.

.. _described here: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.data_vars.html

.. code-block:: python

    In [5]: filename = '3B-MO.MS.MRG.3IMERG.20160101-S000000-E235959.01.V03D.HDF5'
    In [6]: es = load_array(filename)
    In [7]: es.data_vars
    Out[7]:
    Data variables:
        band_0   (y, x) int16 -9999 -9999 -9999 -9999 -9999 -9999 -9999 -9999 ...
        band_1   (y, x) float32 -9999.9 -9999.9 -9999.9 -9999.9 -9999.9 -9999.9 ...
        band_2   (y, x) int16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
        band_3   (y, x) float32 -9999.9 -9999.9 -9999.9 -9999.9 -9999.9 -9999.9 ...

    In [8]: es.band_order
    Out[8]: ['band_0', 'band_1', 'band_2', 'band_3']

.. _common-elm-store-transformations:

Common ``ElmStore`` Transformations
-----------------------------------

.. _transform-flatten:

**Flatten**

``elm.pipeline.steps.Flatten`` will convert an ``ElmStore`` of 2-D rasters in bands (each band as a ``DataArray`` ) to an ``ElmStore`` with a single ``DataArray`` called ``flat``.  *Note: ``elm.pipeline.steps.Flatten()`` must be included in a ``Pipeline`` before scikit-learn based transforms on an ``ElmStore``, where the scikit-learn transforms expect a 2-D array.

Here is an example of ``Flatten`` that continues the example above that defined ``sampler``, a function returning a random ``ElmStore`` of 2-D ``DataArray`` bands:

.. code-block:: python

    es = sampler()
    X_2d, y, sample_weight = steps.Flatten().fit_transform(es)

    In [17]: X_2d.flat
    Out[17]:
    <xarray.DataArray 'flat' (space: 1000000, band: 4)>
    array([[ 1.13465339, -0.1533531 ,  1.72809878, -0.7746218 ],
           [-0.12378515, -1.72588715,  0.07752273, -1.19004227],
           [ 2.16456385, -0.58083733,  0.03706811,  0.26274225],
           ...,
           [ 0.45586256, -1.87248571,  1.27793313,  0.19892153],
           [ 2.11702651, -0.05300853, -0.92923591, -1.07152977],
           [-0.10245425, -1.27150399, -1.48745754,  1.00873062]])
    Coordinates:
      * space    (space) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ...
      * band     (band) <U2 'b1' 'b2' 'b3' 'b4'
    Attributes:
        old_dims: [('y', 'x'), ('y', 'x'), ('y', 'x'), ('y', 'x')]
        _dummy_canvas: True
        canvas: Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=100000, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=-9909.900000000001))
        old_canvases: [Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=100000, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=-9909.900000000001)), Canvas(geo_transform=(-180, 0.1, 0, 90, 0, -0.1), buf_xsize=10, buf_ysize=100000, dims=('y', 'x'), ravel_order='C', zbounds=None, tbounds=None, zsize=None, tsize=None, bounds=BoundingBox(left=-180.0, bottom=90.0, right=-179.1, top=-9909.900000...
        flatten_data_array: True
        band_order: ['b1', 'b2', 'b3', 'b4']

.. _transform-inverseflatten:

**InverseFlatten**

``elm.pipeline.steps.InverseFlatten`` converts an ``ElmStore`` that is flattened (typically the output of :ref:`transform-flatten` above) back to separate 2-D raster bands.

.. code-block:: python

    es = sampler()
    X_2d, y, sample_weight = steps.Flatten().fit_transform(es)
    restored, _, _ = steps.InverseFlatten().fit_transform(X_2d)
    np.all(restored.b1.values == es.b1.values)

.. _transform-dropnarows:

**DropNaRows**

``elm.pipeline.steps.DropNaRows`` is a transformer that will drop any null rows from an ``ElmStore`` that has a ``DataArray`` called ``flat`` (see :ref:`transform-flatten`).  It drops the null rows while keeping metadata to allow :ref:`transform-inverseflatten` in :doc:`predict_many<predict-many>` . An example usage of ``DropNaRows`` is given in the :doc:`K-Means LANDSAT ``elm`` introduction<cluster_example>`

Here is an example of using ``DropNaRows`` with the ``sampler`` function defined above.

.. code-block:: python

    es = sampler()
    X_2d, _, _ = steps.Flatten().fit_transform(es)
    X_2d.flat.values[:2, :] = np.NaN
    X_no_na, _, _ = steps.DropNaRows().fit_transform(X_2d)
    assert X_no_na.flat.shape[0] == X_2d.flat.shape[0] - 2
    restored = inverse_flatten(X_no_na)
    assert restored.b1.shape == es.b1.shape
    val = restored.b1.values
    assert val[np.isnan(val)].size == 2

.. _transform-agg:

**Agg**

Aggregation along a dimension can be done with ``elm.pipeline.steps.Agg``, referencing either a ``dim`` or ``axis`` .

.. code-block:: python

    In [44]: es = sampler()

    In [45]: agged, _, _ = steps.Agg(dim='y', func='median').fit_transform(es)

    In [46]: agged
    Out[46]:
    ElmStore:
    <elm.ElmStore>
    Dimensions:  (x: 10)
    Coordinates:
      * x        (x) int64 0 1 2 3 4 5 6 7 8 9
    Data variables:
        b1       (x) float64 -0.00231 -0.00294 -0.002797 0.002472 -0.006088 ...
        b2       (x) float64 8.965e-06 0.0001929 -0.007133 0.001447 -0.001846 ...
        b3       (x) float64 -0.0009686 -0.003632 -0.0007322 -0.002221 -0.0039 ...
        b4       (x) float64 0.00667 0.001018 0.002702 0.009274 0.001481 ...
    Attributes:
        _dummy_canvas: True
        band_order: ['b1', 'b2', 'b3', 'b4']

In the example above, ``median`` could have been replaced by any of the following:

 * ``all``
 * ``any``
 * ``argmax``
 * ``argmin``
 * ``max``
 * ``mean``
 * ``median``
 * ``min``
 * ``prod``
 * ``sum``
 * ``std``
 * ``var``

Read more on the implementation of the functions above in the `xarray.DataArray methods here`_.

.. _xarray.DataArray methods here: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html

.. _elm-store-metadata:

``ElmStore`` and Metadata
-------------------------

This section describes ``elm`` functions useful for deriving information from file metadata.

.. set-na-from-meta:

**set_na_from_meta**: This function searches the ``attrs`` of each ``DataArray`` in an ``ElmStore`` or ``xarray.Dataset`` and sets ``NaN`` values in each ``DataArray`` where metadata indicates it is necessary.  Currently ``set_na_from_meta`` searches ``attrs`` for the following keys using a case-, space- and punctuation-insenstive regular expression:

 * ``missing_value``: Any values in the ``DataArray`` equal to the missing value will be set to ``NaN``.
 * ``valid_range`` and ``invalid_range``: If ``attrs`` have a key like ``valid_range`` or ``invalid_range``, the function will check to see if it is a sequence of length 2 or a string that can be split on comma or spaces to form a sequnce of length 2.  If a sequence of length 2, then the invalid / valid ranges will be used to set ``NaN`` values appropriately.

.. code-block:: python

    from earthio.tests.util import HDF4_FILES
    from earthio import load_array, set_na_from_meta
    es = load_array(HDF4_FILES[0])
    set_na_from_meta(es) # modifies ElmStore instance in place

.. _meta-is-day:

**meta_is_day**: This function takes a single argument, a dict that is typically the ``attrs`` of an ``ElmStore``, and searches for keys/values indicating whether the ``attrs`` correspond to a day or night sample.

.. code-block:: python

    from earthio.tests.util import HDF4_FILES
    from earthio import load_array
    from earthio.metadata_selection import example_meta_is_day
    from scipy.stats import describe
    es3 = load_array(HDF4_FILES[0])
    es3.DayNightFlag # prints "Day"
    meta_is_day(es3) # prints True
