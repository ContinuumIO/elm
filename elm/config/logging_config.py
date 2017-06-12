from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import sys

ELM_LOGGING_LEVEL = os.environ.get('ELM_LOGGING_LEVEL', 'INFO')
ELM_LOGGING_LEVEL = getattr(logging, ELM_LOGGING_LEVEL)
logger = logging.getLogger(__name__.partition('.')[0])

LOGFILE = os.environ.get('ELM_LOG_FILE', 'logfile.txt')

_has_been_called = False
def init_logging(logfile=LOGFILE):
    global _has_been_called
    if _has_been_called:
        return
    _has_been_called = True
    ELM_LOGGING_LEVEL = os.environ.get("ELM_LOGGING_LEVEL", "INFO")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s')
    fh = logging.FileHandler(logfile, mode='a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.setLevel(getattr(logging, ELM_LOGGING_LEVEL))
    logger.addHandler(ch)
    logger.addHandler(fh)


init_logging()
