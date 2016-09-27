
Feature selection config format
========

Feature selection can use scikit-learn feature selectors or custom functions.  Within `feature_selection`, each key / value is a feature selector identifier and dict specification.  In each of those dicts, the following keys can be used:
 * `selection`: The feature selector, e.g. `"sklearn.feature_selection:SelectPercentile"` to be imported
 * `kwargs`: Key word arguments to `selection` func
 * `scoring`: Scoring method, typically an attribute of `sklearn.feature_selection` such as `f_classif`
 * `choices`: Column (band) names that are passed to the selector (default: all columns)

.. code-block:: bash 

    feature_selection: {
      top_80_percent: {
        selection: "sklearn.feature_selection:SelectPercentile",
        kwargs: {percentile: 80},
        scoring: f_classif,
        choices: all,
      }
    }
