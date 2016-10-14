
import inspect


def get_args_kwargs_defaults(func):
    '''Get the default kwargs spec of a function '''
    sig = inspect.signature(func)
    params = sig.parameters
    kwargs = {}
    args = []
    takes_variable_keywords = None
    for k, v in params.items():
        if v.default != inspect._empty:
            kwargs[k] = v.default
        else:
            args.append(k)
        if v.kind == 4:
            #<_ParameterKind.VAR_KEYWORD: 4>
            takes_variable_keywords = k
    return args, kwargs, takes_variable_keywords


def filter_kwargs_to_func(func, **kwargs):
    arg_spec, kwarg_spec, takes_variable_keywords = get_args_kwargs_defaults(func)
    new = {}
    for k,v in kwargs.items():
        if k in kwarg_spec:
            new[k] = v
    if takes_variable_keywords:
        new[takes_variable_keywords] = {k: v for k,v in kwargs.items()
                                        if not k in new}
    return new
