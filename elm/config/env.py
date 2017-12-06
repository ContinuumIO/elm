from __future__ import absolute_import, division, print_function

'''This module parses environment variables used by elm.

See also elm/config/defaults/environment_vars_spec.yaml
which names all the environment variables, their types, and
their defaults.

'''
import os

from elm.config.util import ElmConfigError, read_from_egg

ENVIRONMENT_VARS_SPEC = read_from_egg(os.path.join('defaults',
                                                   'environment_vars_spec.yaml'))

def process_int_env_var(env_var_name, default='0', required=False):
    '''Process an env var which must be an integer'''
    val = os.environ.get(env_var_name, default)
    try:
        val = bool(int(val))
    except Exception as e:
        if required:
            raise ElmConfigError('Expected env var {} to be parsed '
                                   'as int (got {})'.format(env_var_name, val))
        val = bool(val)
    return val


def process_str_env_var(env_var_name, expanduser=False,
                        default='', required=False, choices=None):
    '''Process a string environment variable that may be a path or have
    fixed choices'''
    val =  os.environ.get(env_var_name, default)
    if choices:
        if not val in choices:
            raise ElmConfigError('Expected env var {} to be '
                                   'in choices {} '
                                   '(go {}'
                                   ')'.format(env_var_name, choices, val))
    if required and (not val and not default):
        raise ElmConfigError('Expected env var {} to be '
                               'defined'.format(env_var_name))
    elif not val:
        val = default
    if expanduser:
        val = os.path.expanduser(val)
    return val


def parse_env_vars():
    '''Process the environment vars specifications
    in defaults/environment_vars_specs.yaml, making sure
    that required env vars are present, that they have values
    among the available choices, and that they are of the correct
    data type after parsing

    Returns:
        elm_env_vars: dict of environment vars relative to elm
    '''
    int_fields_specs = ENVIRONMENT_VARS_SPEC['int_fields_specs']
    str_fields_specs = ENVIRONMENT_VARS_SPEC['str_fields_specs']
    elm_env_vars = {}
    for item in int_fields_specs:
        val = process_int_env_var(item['name'],
                                  default=item.get('default', None),
                                  required=item.get('required', False))
        elm_env_vars[item['name']] = val
    for item in str_fields_specs:
        val = process_str_env_var(item['name'],
                                  expanduser=item.get('expanduser', None),
                                  default=item.get('default', None),
                                  required=item.get('required', False),
                                  choices=item.get('choices', []))
        elm_env_vars[item['name']] = val
    for f in ('DASK_PROCESSES', 'DASK_THREADS'):
        if not elm_env_vars.get(f):
            try:
                import psutil
            except ImportError:
                psutil = None
            cpu_count = getattr(os, 'cpu_count', getattr(psutil, 'cpu_count', None))
            elm_env_vars[f] = cpu_count() if cpu_count else 4
    example_path = elm_env_vars['ELM_EXAMPLE_DATA_PATH']
    if not example_path or not os.path.exists(example_path):
        elm_env_vars['ELM_HAS_EXAMPLES'] = False
    else:
        elm_env_vars['ELM_HAS_EXAMPLES'] = True
    return elm_env_vars
