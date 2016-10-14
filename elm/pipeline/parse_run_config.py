from collections import defaultdict
import copy
import logging

import dask

from elm.config import ConfigParser
from elm.model_selection.evolve import ea_setup
from elm.pipeline.ensemble import ensemble
from elm.pipeline.evolve_train import evolve_train
from elm.pipeline.predict_many import predict_many
from elm.sample_util.sample_pipeline import create_sample_from_data_source
from elm.pipeline.transform import get_new_or_saved_transform_model

logger = logging.getLogger(__name__)


def parse_run_config(config, client):
    '''Run all steps of a config's "pipeline"
    Parameters:
        config: elm.config.ConfigParser instance
        client: Executor/client from Distributed, thread pool
                or None for serial evaluation
    '''
    config.to_pipeline_func(client)
    return 0

