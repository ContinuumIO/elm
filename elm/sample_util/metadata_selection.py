from collections import OrderedDict

from gdalconst import GA_ReadOnly
import gdal
import logging
import pandas as pd
import re

from elm.readers.util import BandSpec
from elm.model_selection.util import get_args_kwargs_defaults
from elm.sample_util.util import InvalidSample

logger = logging.getLogger(__name__)


DAY_NIGHT_WORDS = ('day', 'night')
FLAG_WORDS = ('flag', 'indicator')
DAY_NIGHT = []
for f in FLAG_WORDS:
    w1 = "".join(DAY_NIGHT_WORDS)
    w2 = "".join(DAY_NIGHT_WORDS[::-1])
    w3, w4 = DAY_NIGHT_WORDS
    for w in (w1, w2, w3, w4):
        w5, w6 = f + w, w + f
        DAY_NIGHT.extend((w5, w6,))

def _strip_key(k):
    if isinstance(k, str):
        for delim in ('.', '_', '-', ' '):
            k = k.lower().replace(delim,'')
    return k

def match_meta(meta, band_spec):
    '''
    Parmeters
    ---------
    meta: dataset meta information object
    band_spec: BandSpec object

    Returns
    -------
    boolean of whether band_spec matches meta
    '''
    if not isinstance(band_spec, BandSpec):
        raise ValueError('band_spec must be elm.readers.BandSpec object')

    for mkey in meta:
        key_re_flags = [getattr(re, att)
                        for att in (band_spec.key_re_flags or [])]
        value_re_flags = [getattr(re, att)
                        for att in (band_spec.value_re_flags or [])]

        if bool(re.search(band_spec.search_key, mkey, *key_re_flags)):
            if bool(re.search(band_spec.search_value, meta[mkey], *value_re_flags)):
                return True
    return False


def example_meta_is_day(filename, d):

    dicts = []
    for k, v in d.items():
        if isinstance(v, dict):
            dicts.append(v)
            continue
        key2 = _strip_key(k)
        if key2 in DAY_NIGHT:
            dayflag = 'day' in key2
            nightflag = 'night' in key2
            if dayflag and nightflag:
                value2 = _strip_key(v)
                return 'day' in value2
            elif dayflag or nightflag:
                return bool(v)
    if dicts:
        return any(example_meta_is_day(filename, d2) for d2 in dicts)
    return False


