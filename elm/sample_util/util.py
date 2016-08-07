
def bands_as_columns(func):
    def new_func(es, *args, **kwargs):
        from elm.pipeline.sample_pipeline import flatten_cube
        return func(flatten_cube(es), *args, **kwargs)
    return new_func

