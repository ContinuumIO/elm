## Reading HDF4 / 5, GeoTiff, and NetCDF Files

### Through the elm pipeline config system

 * Include a `readers` dictionary that is a mapping of identifiers to file array and meta data loading functions.  Example file loaders from elm.readers that can be included:

```
readers: {
  hdf4_example: {
    load_array: "elm.readers.hdf4:load_hdf4_array",
    load_meta: "elm.readers.hdf4:load_hdf4_meta"
  },
  tif_example: {
    load_array: "elm.readers.tif:load_dir_of_tifs_array",
    load_meta: "elm.readers.tif:load_dir_of_tifs_meta",
  },
  hdf5_example: {
    load_array: "elm.readers.hdf5:load_hdf5_array",
    load_meta: "elm.readers.hdf5:load_hdf5_meta",
  },
  netcdf_example: {
    load_array: "elm.readers.netcdf:load_netcdf_array",
    load_meta: "elm.readers.netcdf:load_netcdf_meta",
  }
}
```
The dictionary keys above like `hdf5_example` are referenced in the `data_sources` section as a `reader`.

 * Here is an example a `data_sources` section of a config:
```
data_sources: {
 hdf5_precip_hourly: {
   reader: hdf5_example,
   sample_from_args_func: "elm.sample_util.samplers:image_selection",
   band_specs: [{search_key: sub_dataset_name,
                 search_value: /precipitation,
                 name: band_1,
                 key_re_flags: [],
                 value_re_flags: [],
                 meta_to_geotransform: "elm.sample_util.metadata_selection:grid_header_hdf5_to_geo_transform",
                 stored_coords_order: ["x", "y"]}],
   selection_kwargs: {
    file_pattern: 3B-MO.MS.MRG.3IMERG.20160101-S000000-E235959.01.V03D.HDF5,
    top_dir: "env:ELM_EXAMPLE_DATA_PATH",
   },
   batch_size: 1440000,
   sample_args_generator: iter_files_recursively,
  },
 NPP_DSRF1KD_L2GD: {
  reader: hdf4_example,
  sample_from_args_func: "elm.sample_util.samplers:image_selection",
  band_specs: [{search_key: long_name, search_value: "Band 1 ", name: band_1},
  {search_key: long_name, search_value: "Band 2 ", name: band_2},
  {search_key: long_name, search_value: "Band 3 ", name: band_3},
  {search_key: long_name, search_value: "Band 4 ", name: band_4},
  {search_key: long_name, search_value: "Band 5 ", name: band_5},
  {search_key: long_name, search_value: "Band 6 ", name: band_6},
  {search_key: long_name, search_value: "Band 7 ", name: band_7},
  {search_key: long_name, search_value: "Band 9 ", name: band_9},
  {search_key: long_name, search_value: "Band 10 ", name: band_10},
  {search_key: long_name, search_value: "Band 11 ", name: band_11}],
  sample_args_generator: iter_files_recursively,
  selection_kwargs: {
    top_dir: "env:ELM_EXAMPLE_DATA_PATH",
    file_pattern: .*\.hdf,
    metadata_filter: "elm.sample_util.metadata_selection:example_meta_is_day",
  },
  batch_size: 1440000,
 },
}

```

The config above shows the parts of the each data source:
 * `reader` points to the file metadata and array loading functions named elsewhere
 * `band_specs` is a list as long as the number of bands to extract from each file.  Within each `band_spec`, `search_key` defines the key to search in the band metadata to find a matching band and `search_value` is what it needs to match while `name` is the name the band will be given in the ElmStore (xarray Dataset) that is created.  A `band_spec` may also include `key_re_flags: [IGNORECASE]` or `value_re_flags: [IGNORECASE]` to do the corresponding regular expressions for band matching in a case-insensitive way. In the example above, the first data source gets 1 band and the second data source has 10 bands. The key / value `meta_to_geotransform: "elm.sample_util.metadata_selection:grid_header_hdf5_to_geo_transform"` specifies to take the geo transform from a metadata reader rather than to rely on the default `GetGeoTransform` of GDAL.  `stored_coords_order` defaults to `["y", "x"]`, a typical "north-up" image, but can be used with `["x", "y"]` for images stored in a transpose of this pattern: `stored_coords_order: ["x", "y"]`
 * `selection_kwargs` are keyword arguments passed to the file selection logic, i.e. `sample_args_generator`, with typical keywords being `file_pattern` and `top_dir` (search recursively within `top_dir` matching file pattern)
 * `sample_args_generator` names the a generator of arguments from the `sample_args_generators` section of the config, which in this case is to iterate files recursively over the `top_dir`.

