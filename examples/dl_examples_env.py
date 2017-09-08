#!/usr/bin/env python3


import sys
import os
import requests
import shutil
import tempfile
import yaml
from collections import OrderedDict


ENV_SETUP_FILES = {
    'download_sample_data.py': 'https://raw.githubusercontent.com/bokeh/datashader/master/examples/download_sample_data.py',
    'datasets.yml': 'https://raw.githubusercontent.com/bokeh/datashader/master/examples/datasets.yml',
    'environment.yml': 'https://raw.githubusercontent.com/bokeh/datashader/master/examples/environment.yml',
}


def dl_content(dst_fpath, url, tmpdir):
    """Download content from url and save to dst_path."""
    with open(os.path.join(tmpdir, dst_fpath), 'wb') as f:
        resp = requests.get(url)
        print('Downloading "{}"...'.format(url))
        f.write(resp.content)


def update_env_yml(fpath, tmpdir):
    """Update environment.yml in tmpdir, and write the new version to current
    directory.
    """
    print('Updating {} for elm...'.format(fpath))

    # Preserve yaml ordering: https://stackoverflow.com/a/21048064
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())
    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))
    yaml.add_representer(OrderedDict, dict_representer)
    yaml.add_constructor(_mapping_tag, dict_constructor)

    with open(os.path.join(tmpdir, fpath), 'r') as fp:
        env_data = OrderedDict(yaml.safe_load(fp))
    env_data['name'] = 'elm-examples'
    env_data['channels'].extend(['elm', 'elm/label/dev', 'numba'])

    deps_to_add = ['earthio', 'elm','fastparquet', 'pyarrow']
    deps = env_data['dependencies']
    pip_deps_idx = None
    for deps_idx, elt in enumerate(deps):
        if isinstance(elt, dict) and 'pip' in elt:
            pip_deps_idx = deps_idx
        elif elt.startswith('python='):
            env_data['dependencies'][deps_idx] = 'python=3.5'
    if pip_deps_idx is not None:
        env_data['dependencies'] = deps[:pip_deps_idx] + deps_to_add + deps[pip_deps_idx:]
    else:
        env_data['dependencies'].extend(deps_to_add)

    with open(fpath, 'w') as fp:
        yaml.dump(env_data, fp, default_flow_style=False)


def main(argv):
    tmpdir = tempfile.mkdtemp()
    try:
        # Download the environment setup files
        for dst_fpath, url in ENV_SETUP_FILES.items():
            dl_content(dst_fpath, url, tmpdir)

        # Update the environment.yml for elm's examples
        assert 'environment.yml' in ENV_SETUP_FILES
        update_env_yml('environment.yml', tmpdir)
        for f in (set(ENV_SETUP_FILES) - set(('environment.yml',))):
            shutil.copy(os.path.join(tmpdir, f), f)
        print('Finished downloading. Next steps:\n'
              '    $ conda env create -f environment.yml\n'
              '    $ source activate elm-examples\n'
              '    $ python download_sample_data.py')
    finally:
        shutil.rmtree(tmpdir)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
