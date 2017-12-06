from xarray_filters.pipeline import Generic, Step

class ChooseWithPreproc(Step):

    estimator = None
    trans_if = None
    trans = None
    run = True

    def _pre_trans(self, X):
        if trans_if and trans_if(self):
            return self.trans(X)
        return X

    def transform(self, X, y=None, **kw):
        if not self.run:
            return X
        X = self._pre_trans(X)
        return self.estimator.transform(X, y=y, **kw)

    def fit_transform(self, X, y=None, **kw):
        if not self.run:
            return X
        X = self._pre_trans(X)
        return self.estimator.fit_transform(X, y=y, **kw)

    def fit(self, X, y=None, **kw):
        if not self.run:
            return X
        X = self._pre_trans(X)
        return self.estimator.fit(X, y=y, **kw)