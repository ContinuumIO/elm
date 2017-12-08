from xarray_filters.pipeline import Generic, Step

class ChooseWithPreproc(Step):

    estimator = None
    use_transform = None
    run = True

    def _pre_trans(self, X):
        X, y = X
        if self.use_transform:
            return self.use_transform(X, y=y)
        return X

    def transform(self, X, y=None, **kw):
        if not self.run:
            return X,y
        X = self._pre_trans(X)
        return self.estimator.transform(X, y=y, **kw)

    def fit_transform(self, X, y=None, **kw):
        if not self.run:
            return X,y
        X = self._pre_trans(X)
        return self.estimator.fit_transform(X, y=y, **kw), y

    def fit(self, X, y=None, **kw):
        if not self.run:
            return X,y
        X = self._pre_trans(X)
        return self.estimator.fit(X, y=y, **kw), y
