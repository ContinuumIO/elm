from __future__ import absolute_import, division, print_function, unicode_literals

from importlib import import_module
'''
----------------------

``elm.pipeline.util``
~~~~~~~~~~~~~~~~~~~~~

Internal helpers for elm.pipeline
'''

_next_idx = -1

def _next_name(token):
    '''name in a dask graph'''
    global _next_idx
    n = '{}-{}'.format(token, _next_idx)
    _next_idx += 1
    return n



