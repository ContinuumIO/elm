import array
from collections import namedtuple
import copy
import inspect

from deap import creator, base, tools
from deap.tools.emo import selNSGA2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from elm.config import delayed
from elm.model_selection.util import get_args_kwargs_defaults

def no_selection(models, **kwargs):
    return models

def base_selection(models, **kwargs):
    pass


