'''
----------------------------------

``elm.sample_util.meta_selection``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
from collections import OrderedDict

from gdalconst import GA_ReadOnly
import gdal
import logging
import pandas as pd
import re

from elm.readers.util import BandSpec
from elm.config.func_signatures import get_args_kwargs_defaults

logger = logging.getLogger(__name__)

def _strip_key(k):
    if isinstance(k, str):
        for delim in ('.', '_', '-', ' '):
            k = k.lower().replace(delim,'')
    return k

def match_meta(meta, band_spec):
    '''
    Parmeters:
        :meta: dataset meta information object
        :band_spec: BandSpec object

    Returns:
        :boolean: of whether band_spec matches meta

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


def meta_is_day(attrs):
    '''Helper to find day/ night flags in nested dict

    Parmeters:
        :d: dict

    Returns:
        :True: if day, **False** if night, else None

    '''
    dicts = []
    for k, v in attrs.items():
        if isinstance(v, dict):
            dicts.append(v)
            continue
        key2 = _strip_key(k)
        dayflag = 'day' in key2
        nightflag = 'night' in key2
        if dayflag and nightflag:
            value2 = _strip_key(v)
            return 'day' in value2.lower()
        elif dayflag or nightflag:
            return bool(v)
    if dicts:
        return any(meta_is_day(d2) for d2 in dicts)
    return False
