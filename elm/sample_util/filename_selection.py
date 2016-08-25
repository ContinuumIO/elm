import re

from elm.sample_util.band_selection import _select_from_file_base


def _filename_filter(filename, search=None, func=None):
    if search is None and func is None:
        return True
    if search is not None:
        keep = re.search(search, filename)
    else:
        keep = True
    if func is None:
        return keep
    else:
        return func(filename) and keep


def include_file(filename, band_specs, **selection_kwargs):
    selection_kwargs['dry_run'] = True
    return _select_from_file_base(filename, band_specs, **selection_kwargs)

def get_generated_args(filenames_gen, band_specs, no_file_open=False, **selection_kwargs):
    return tuple(f for f in filenames_gen(**selection_kwargs)
                 if include_file(f, band_specs, no_file_open=no_file_open, **selection_kwargs))

