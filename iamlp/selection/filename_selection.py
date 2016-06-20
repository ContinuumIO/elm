import re

from iamlp.selection.band_selection import _select_from_file_base


def _filter_on_filename(filename, search=None, func=None):
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


def include_file(filename, band_specs, no_file_open=True, **selection_kwargs):
    if no_file_open:
        return _filter_on_filename(filename)
    selection_kwargs['dry_run'] = True
    return _select_from_file_base(filename, band_specs, **selection_kwargs)

def get_included_filenames(filenames_gen, band_specs, no_file_open=True, **selection_kwargs):
    return tuple(f for f in filenames_gen()
                 if include_file(f, band_specs, **selection_kwargs))

