import inspect

def get_args_kwargs_defaults(func):
    '''Get the default kwargs spec of a function '''
    sig = inspect.signature(sc.MiniBatchKMeans)
    params = sig.parameters
    kwargs = {}
    args = []
    for k, v in params.items():
        if v.default != inspect._empty:
            kwargs[k] = v.default
        else:
            args.append(k)
    return args, kwargs