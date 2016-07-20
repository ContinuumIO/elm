from io import StringIO
import os
import shutil
import subprocess as sp
import tempfile

import pytest
import yaml

from elm.pipeline import pipeline
from elm.config import DEFAULTS, DEFAULTS_FILE

@pytest.yield_fixture
def scenario_fixture():
    tmp1, tmp2, tmp3 = (tempfile.mkdtemp() for _ in range(3))
    old1, old2 = os.environ.get('ELM_PICKLE_PATH') or '', os.environ.get('ELM_PREDICT_PATH') or ''
    os.environ['ELM_PICKLE_PATH'] = tmp1
    os.environ['ELM_PREDICT_PATH'] = tmp2
    yield (tmp1, tmp2, tmp3)
    os.environ['ELM_PICKLE_PATH'] = old1
    os.environ['ELM_PREDICT_PATH'] = old2
    for tmp in (tmp1, tmp2):
        if os.path.exists(tmp):
            shutil.rmtree(tmp)


def tst_one_config(pickle_path, predict_path, config=None, cwd=None):

    config_str = yaml.dump(config or DEFAULTS)
    config_filename = os.path.join(cwd, 'config.yaml')
    with open(config_filename, 'w') as f:
        f.write(config_str)
    proc = sp.Popen(['elm-main',
                      '--config',
                      config_filename,
                      '--echo-config'],
                     cwd=cwd,
                     stdout=sp.PIPE,
                     stderr=sp.STDOUT,
                     env=os.environ)
    r = proc.wait()
    log = proc.stdout.read().decode()
    print(log)
    if r != 0:
        raise ValueError(log)
    return log

def test_default_config(scenario_fixture):
    pickle_path, predict_path, cwd = scenario_fixture
    out = tst_one_config(pickle_path, predict_path, config=DEFAULTS, cwd=cwd)
    len_train, len_predict = map(os.listdir, (pickle_path, predict_path))
    assert len_train
    assert len_predict
    assert 'elm.scripts.main - ok' in out
