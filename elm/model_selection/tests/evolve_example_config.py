CONFIG_STR = '''
ensembles: {
    save_one: {
        saved_ensemble_size: 1,
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
  nothing: [],
}

param_grids: {
  example_param_grid: {
    kmeans__n_clusters: [3,4,5,6,7,8],
    pca__n_components: [2,3,4,5],
    sample_pipeline: [minimal, top_n],
    feature_selection: {top_n: {kwargs: {percentile: [30, 40, 50, 60, 70, 80, 90],}}},
    control: {
      select_method: selNSGA2,
      crossover_method: cxTwoPoint,
      mutate_method: mutUniformInt,
      init_pop: random,
      indpb: 0.5,
      mutpb: 0.9,
      cxpb:  0.3,
      eta:   20,
      ngen:  2,
      mu:    4,
      k:     4,
      early_stop: {abs_change: [10], agg: all},
      # alternatively early_stop: {percent_change: [10], agg: all}
      # alternatively early_stop: {threshold: [10], agg: any}
    }
  }
}

data_sources: {
  synthetic: {
    sample_from_args_func: 'elm.pipeline.tests.util:random_elm_store',
    sampler_args: [],
  }
}

model_scoring: {
  testing_model_scoring: {
    score_weights: [-1],
    scoring: 'elm.model_selection.kmeans:ensemble_kmeans_scoring',
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
- train: kmeans
  param_grid: example_param_grid
  sample_pipeline: minimal

'''
