import os

from iamlp.config.util import IAMLPConfigError, read_from_egg

ENVIRONMENT_VARS_SPEC = read_from_egg(os.path.join('defaults',
                                                   'environment_vars_spec.yaml'))

def process_int_env_var(env_var_name, default='0', required=False):
    val = os.environ.get(env_var_name, default)
    try:
        val = bool(int(val))
    except Exception as e:
        if required:
            raise IAMLPConfigError('Expected {} to be parsed '
                                   'as int (env var {})'.format(val, env_var_name))
        val = bool(val)
    return val

def process_str_env_var(env_var_name, default='', required=False, choices=None):
    val =  os.environ.get(env_var_name, default)
    if choices:
        if not val in choices:
            raise IAMLPConfigError('Expected {} to be '
                                   'in choices {} '
                                   '(env var {}'
                                   ')'.format(val, choices, env_var_name))
    if required and not val:
        raise IAMLPConfigError('Expected env var {} to be '
                               'defined'.format(env_var_name))
    return val

def parse_env_vars():
    int_fields_specs = ENVIRONMENT_VARS_SPEC['int_fields_specs']
    str_fields_specs = ENVIRONMENT_VARS_SPEC['str_fields_specs']
    relevant_env = {}
    for item in int_fields_specs:
        val = process_int_env_var(item['name'],
                                  default=item.get('default', '0'),
                                  required=item.get('required', False))
        relevant_env[item['name']] = val
    for item in str_fields_specs:
        val = process_str_env_var(item['name'],
                                  default=item.get('default', '0'),
                                  required=item.get('required', False),
                                  choices=item.get('choices', []))
        relevant_env[item['name']] = val
    for f in ('DASK_PROCESSES', 'DASK_THREADS'):
        if not relevant_env.get(f):
            relevant_env[f] = os.cpu_count()
    return relevant_env

__all__ = ['ENV']