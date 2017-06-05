from __future__ import absolute_import, division, print_function, unicode_literals

CONFIG_STR = '''
ensembles: {
    save_one: {
        saved_ensemble_size: 1,
    }
}
transform: {
  pca: {
    model_init_class: "sklearn.decomposition:IncrementalPCA",
    model_init_kwargs: {n_components: 2}
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
    func: "elm.sample_util.preproc_scale:require_positive",
    func_kwargs: {small_num: 0.0001},
  },
}

param_grids: {
  example_param_grid: {
    kmeans__n_clusters: [3,4,5,6,7,8],
    pca__n_components: [2,3,4,5],
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
    sampler: 'elm.sample_util.make_blobs:random_elm_store',
    sampler_args: [[["band_1", "band_2", "band_3", "band_4", "band_5"]]],
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
    model_scoring: testing_model_scoring,
  }
}

run:
- {pipeline: [{flatten: C},
    {sklearn_preprocessing: require_positive},
    {sklearn_preprocessing: log10},
    {sklearn_preprocessing: standard},
    {transform: pca},
   ],
   data_source: synthetic,
   train: kmeans,
   param_grid: example_param_grid}

'''
