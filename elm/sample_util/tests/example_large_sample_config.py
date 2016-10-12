CONFIG_STR = '''
readers: {
  hdf4_example: {
     load_array: "elm.readers.hdf4:load_hdf4_array",
     load_meta: "elm.readers.hdf4:load_hdf4_meta"
  },
  dir_of_tifs_reader: {
     load_array: "elm.readers.tif:load_dir_of_tifs_array",
     load_meta: "elm.readers.tif:load_dir_of_tifs_meta"
   },
}
ensembles: {
    save_one: {
        saved_ensemble_size: 4,
        init_ensemble_size: 10,
        partial_fit_batches:  2,
        ngen:  3,
    }
}
transform: {
  pca: {
    model_init_class: "sklearn.decomposition:IncrementalPCA",
    model_init_kwargs: {"n_components": 2},
    ensemble: save_one,
  }
}

sklearn_preprocessing: {
  standard: {
    method: StandardScaler,
    copy: False,
    with_mean: True,
    with_std: True,
  },
  log10: {
    method: FunctionTransformer,
    func: "numpy:log10",
    validate: True,
  },
  require_positive: {
    method: FunctionTransformer,
    func: "elm.sample_util.encoding_scaling:require_positive",
    func_kwargs: {small_num: 0.0001},
  },
}

feature_selection: {
  top_n: {
    selection: "sklearn.feature_selection:SelectPercentile",
    kwargs: {percentile: 80},
    scoring: f_classif,
    choices: all,
  }
}

sample_pipelines: {
  minimal: [
    {flatten: C},
    {sklearn_preprocessing: require_positive},
    {sklearn_preprocessing: log10},
    {sklearn_preprocessing: standard},
  ],
  top_n: [
    {flatten: C},
    {sample_pipeline: minimal},
    {feature_selection: top_n},
  ],
  nothing: [{flatten: C},],
}

data_sources: {
 NPP_DSRF1KD_L2GD: {
  reader: hdf4_example,
  sampler: "elm.sample_util.samplers:image_selection",
  band_specs: [[long_name, "Band 1 ", band_1],
  [long_name, "Band 2 ", band_2],
  [long_name, "Band 3 ", band_3],
  [long_name, "Band 4 ", band_4],
  [long_name, "Band 5 ", band_5],
  [long_name, "Band 7 ", band_7],
  [long_name, "Band 8 ", band_8],
  [long_name, "Band 9 ", band_9],
  [long_name, "Band 10 ", band_10],
  [long_name, "Band 11 ", band_11]],
  args_gen: iter_files_recursively,

  top_dir: "env:ELM_EXAMPLE_DATA_PATH",
  file_pattern: "\\.hdf",

  data_filter: Null,
  metadata_filter: "elm.sample_util.metadata_selection:example_meta_is_day",
  filename_filter: Null,
  geo_filters: {
    include_polys: [],
    exclude_polys: [],
  },
  },
  keep_columns: [],
  batch_size: 1440000,
 },

}

model_scoring: {
  testing_model_scoring: {
    score_weights: [-1],
    scoring: 'elm.model_selection.kmeans:kmeans_aic',
  }
}

train: {
  kmeans: {
    model_init_class: "sklearn.cluster:MiniBatchKMeans",
    model_init_kwargs: {compute_labels: True},
    ensemble: save_one,
    output_tag: kmeans,
    model_scoring: testing_model_scoring,
  }
}
predict: {
  kmeans: {
  }
}

pipeline:
- {data_source: NPP_DSRF1KD_L2GD,
  sample_pipeline: minimal,
  steps: [
    {train: kmeans},
    {predict: kmeans},
    ]
  }
args_gen: {
  tif_file_gen: "elm.readers.local_file_iterators:iter_dirs_of_dirs",
  iter_files_recursively: "elm.readers.local_file_iterators:iter_files_recursively",
}
'''


