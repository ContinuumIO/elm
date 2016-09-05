import yaml
from elm.config import ConfigParser
from elm.sample_util.tests.example_large_sample_config import CONFIG_STR
from elm.pipeline.tests.util import test_one_config as tst_one_config
cwd = '.'
config = yaml.load(CONFIG_STR)
r = tst_one_config(config=config, cwd=cwd)