import os

from elm.config.util import ElmConfigError, read_from_egg

ENVIRONMENT_VARS_SPEC = read_from_egg(os.path.join('defaults',
                                                   'environment_vars_spec.yaml'))

def process_int_env_var(env_var_name, default='0', required=False):
    val = os.environ.get(env_var_name, default)
    try:
        val = bool(int(val))
    except Exception as e:
        if required:
            raise ElmConfigError('Expected env var {} to be parsed '
                                   'as int (got {})'.format(env_var_name, val))
        val = bool(val)
    return val

def process_str_env_var(env_var_name, default='', required=False, choices=None):
    val =  os.environ.get(env_var_name, default)
    if choices:
        if not val in choices:
            raise ElmConfigError('Expected env var {} to be '
                                   'in choices {} '
                                   '(go {}'
                                   ')'.format(env_var_name, choices, val))
    if required and not val:
        raise ElmConfigError('Expected env var {} to be '
                               'defined'.format(env_var_name))
    elif not val:
        val = default
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
        if item['name'] == 'DASK_SCHEDULER':
            required = elm_env_vars['DASK_EXECUTOR'] != 'SERIAL'
        else:
            required = item.get('required', False)
        val = process_str_env_var(item['name'],
                                  default=item.get('default', None),
                                  required=required,
                                  choices=item.get('choices', []))
        elm_env_vars[item['name']] = val
    for f in ('DASK_PROCESSES', 'DASK_THREADS'):
        if not elm_env_vars.get(f):
            elm_env_vars[f] = os.cpu_count()
    example_path = elm_env_vars['ELM_EXAMPLE_DATA_PATH']
    if not example_path or not os.path.exists(example_path):
        elm_env_vars['ELM_HAS_EXAMPLES'] = False
    else:
        elm_env_vars['ELM_HAS_EXAMPLES'] = True
    return elm_env_vars
