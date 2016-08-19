CONFIG_STR = '''
transform: {
  pca: {
    model_init_class: "sklearn.decomposition:IncrementalPCA",
    model_init_kwargs: {"n_components": 2},
    ensemble: no_ensemble,
    model_selection: Null,
    model_scoring: Null,
    data_source: synthetic,
    param_grid: pca_kmeans_small,
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
    {sklearn_preprocessing: require_positive},
    {sklearn_preprocessing: log10},
    {sklearn_preprocessing: standard},
    {transform: pca, method: fit_transform},
  ],
  top_n: [
    {sample_pipeline: minimal},
    {feature_selection: top_n},
    {transform: pca, method: fit_transform},
  ],
}

param_grids: {
  pca_kmeans_small: {
    kmeans__n_clusters: [3,4,5,6,7,8],
    pca__n_components: [2,3,4,5],
    sample_pipeline: [minimal, top_n],
    feature_selection: {top_n: {kwargs: {percentile: [30, 40, 50, 60, 70, 80, 90],}}},
    control: {
      select_method: selNSGA2,
      crossover_method: cxTwoPoint,
      mutate_method: mutShuffleIndexes,
      init_pop: random,
      indpb: 0.2,
      mutpb: 0.4,
      cxpb:  0.3,
      ngen:  2,
      mu:    4,
      k:     4,
    }
  }
}

data_sources:
  synthetic:
    band_specs:
    - [long_name, 'Band 1 ', band_1]
    - [long_name, 'Band 2 ', band_2]
    - [long_name, 'Band 3 ', band_3]
    - [long_name, 'Band 4 ', band_4]
    - [long_name, 'Band 5 ', band_5]
    - [long_name, 'Band 7 ', band_7]
    - [long_name, 'Band 8 ', band_8]
    - [long_name, 'Band 10 ', band_10]
    - [long_name, 'Band 11 ', band_11]
    batch_size: 1440000
    file_pattern: '*.hdf'
    get_weight_func: null
    get_weight_kwargs: {}
    get_y_func: null
    get_y_kwargs: {}
    keep_columns: []
    reader: hdf4-eos
    sample_args_generator: iter_files_recursively
    sample_args_generator_kwargs: {extension: .hdf, top_dir: 'env:ELM_EXAMPLE_DATA_PATH'}
    sample_from_args_func: 'elm.pipeline.tests.util:random_elm_store'
    selection_kwargs:
      data_filter: null
      filename_filter: null
      geo_filters:
        exclude_polys: []
        include_polys: []
      metadata_filter: null
model_scoring: {
    testing_model_scoring: {
        score_weights: [-1],
    }
}
train: {
  kmeans: {
    model_init_class: "sklearn.cluster:MiniBatchKMeans",
    fit_kwargs: {},
    model_init_kwargs: {},
    ensemble: no_ensemble,
    output_tag: kmeans,
    data_source: synthetic,
    keep_columns: [],
    model_scoring: testing_model_scoring,
  }
}

pipeline:
- train: kmeans
  param_grid: pca_kmeans_small
  sample_pipeline: minimal

'''
