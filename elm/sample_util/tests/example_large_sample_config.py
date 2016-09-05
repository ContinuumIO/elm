CONFIG_STR = '''
ensembles: {
    save_one: {
        saved_ensemble_size: 10,
        init_ensemble_size: 600,
        batches_per_gen:  4,
        ngen:  3,
    }
}
transform: {
  pca: {
    model_init_class: "sklearn.decomposition:IncrementalPCA",
    model_init_kwargs: {"n_components": 2},
    ensemble: save_one,
    model_selection: Null,
    model_scoring: Null,
    data_source: synthetic,
    param_grid: example_param_grid,
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
  synthetic: {
    sample_from_args_func: 'elm.pipeline.tests.util:random_elm_store',
    sampler_args: [],
    samples_per_batch: 4,
    random_rows: 8000000,
    n_batches: 10,
  }
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
    fit_kwargs: {},
    model_init_kwargs: {},
    ensemble: save_one,
    output_tag: kmeans,
    data_source: synthetic,
    keep_columns: [],
    model_scoring: testing_model_scoring,
  }
}

pipeline:
- {data_source: synthetic,
  sample_pipeline: minimal,
  steps: [
    {train: kmeans},
    ]
  }
'''


