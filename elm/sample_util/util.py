
def bands_as_columns(func):
    '''Decorator to require that an ElmStore is flattened
    to 2-d (bands as columns)'''
    def new_func(es, *args, **kwargs):
        from elm.pipeline.sample_pipeline import flatten_cube
        return func(flatten_cube(es), *args, **kwargs)
    return new_func

