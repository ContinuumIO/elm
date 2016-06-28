import array
from collections import namedtuple
import copy
import inspect

from deap import creator, base, tools
from deap.tools.emo import selNSGA2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from iamlp.config import delayed
from iamlp.model_selection.util import get_args_kwargs_defaults


def base_selection(models,
                 fit_attr,
                 model_init_class,
                  model_init_kwargs,
                  weights=None):
    pass


