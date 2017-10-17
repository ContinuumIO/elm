class SpecMixinBaseEstimator:

    _root = 'elm.pipeline.steps.{}'
    @property
    def spec(self):
        _cls = getattr(self, '_cls', None)
        if not _cls:
            _cls = self.__class__
        name = _cls.__name__
        module = _cls.__module__.split('.')[1]
        return dict(name=_cls.__name__,
                    module=self._root.format(module),
                    params=self.get_params())

    @classmethod
    def from_spec(self, spec):
        modul, name, params = spec['module'], spec['name'], spec['params']
        parts = modul.split('.')
        elm = '.'.join(parts[:-1])
        sk_module = __import__(elm, globals(), locals())
        for p in parts[1:]:
            sk_module = getattr(sk_module, p)
        return getattr(sk_module, name)(**params)


class PipelineSpecMixin(SpecMixinBaseEstimator):

    @property
    def spec(self):
        steps = [[name, step.spec] for name, step in self.steps]
        spec = super(PipelineSpecMixin, self).spec
        spec['steps'] = steps
        return spec

    @classmethod
    def from_spec(self, spec):
        spec = spec.copy()
        from_spec = super(PipelineSpecMixin, self).from_spec
        steps = [[name, from_spec(spec)] for name, spec in spec.pop('steps')]
        return super(PipelineSpecMixin, self).from_spec(**spec)




