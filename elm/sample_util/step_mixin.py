

class StepMixin(object):

    _sp_step = None
    _func = None
    _required_kwargs = None
    _context = 'sample pipeline step'

    def __init__(self, *args, func=None, **kwargs):
        self._args = args
        self._kwargs = kwargs

        self.func = (func or self._func)
        if self.func:
            self.func = self.func(*args, **kwargs)
        self._validate_init_base()
        if callable(getattr(self, '_validate_init', None)):
            self._validate_init(*self._args, **self._kwargs)

    def get_params(self, **kwargs):
        func = getattr(self.func, 'get_params', None)
        if func:
            return func(**kwargs)
        return self._kwargs

    def _validate_init_base(self):
        if self._sp_step is None:
            raise ValueError('Expected inheriting class to define _sp_step')

    _filter_kw = None

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        ft = getattr(self.func, 'fit_transform', None)
        dot_transform = False
        if not callable(ft):
            ft = getattr(self.func, 'fit', None)
            if ft is not None:
                dot_transform = True
            else:
                ft = self.func
        kw = kwargs.copy()
        if sample_weight is not None:
            kw['sample_weight'] = sample_weight
        if callable(self._filter_kw):
            args, kwargs = self._filter_kw(ft, X, y=y, **kw)
        else:
            args, kwargs = (X, y), kw
        output = ft(*args, **kwargs)
        if dot_transform:
            output = output.transform(X.flat.values)
        return _split_pipeline_output(output, X, y, sample_weight,
                                      getattr(self, '_context', ft))

    def __repr__(self):
        name = self.__class__.__name__
        params = ('{}: {}'.format(k, repr(v))
                  for k, v in sorted(self.get_params().items()))
        return '<elm.steps.{}>:\n\t{}'.format(name, '\n\t'.join(params))

    __str__ = __repr__

