from __future__ import absolute_import, division, print_function, unicode_literals

'''
----------------------------

``earthio.filters.config.func_signatures``

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import inspect
import sys

__all__ = ['get_args_kwargs_defaults', 'filter_kwargs_to_func']

def get_args_kwargs_defaults(func):
    '''Get the required args, defaults, and var keywords of func

    Parameters:
        :func: callable
    Returns:
        :(args, kwargs, takes_var_keywords): where args are names /
        of required args, kwargs are keyword args with defaults, and
        takes_var_keywords indicates whether func has a \*\*param
     '''
    if hasattr(inspect, 'signature'):
        sig = inspect.signature(func) # Python 3
        empty = inspect._empty
    else:
        import funcsigs
        sig = funcsigs.signature(func) # Python 2
        empty = funcsigs._empty
    params = sig.parameters
    kwargs = {}
    args = []
    takes_variable_keywords = None
    for k, v in params.items():
        if v.default != empty:
            kwargs[k] = v.default
        else:
            args.append(k)
        if v.kind == 4:
            #<_ParameterKind.VAR_KEYWORD: 4>
            takes_variable_keywords = k
    return args, kwargs, takes_variable_keywords

def filter_kwargs_to_func(func, **kwargs):
    '''Remove keys/values from kwargs if cannot be passed to func'''
    arg_spec, kwarg_spec, takes_variable_keywords = get_args_kwargs_defaults(func)
    new = {}
    for k,v in kwargs.items():
        if k in kwarg_spec:
            new[k] = v
    if takes_variable_keywords:
        new[takes_variable_keywords] = {k: v for k,v in kwargs.items()
                                        if not k in new}
    return new
