import re

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
    from elm.sample_util.band_selection import _select_from_file_base
    selection_kwargs['dry_run'] = True
    return _select_from_file_base(filename, band_specs, **selection_kwargs)

def get_generated_args(filenames_gen, band_specs, sampler_func, **selection_kwargs):
    from elm.sample_util.band_selection import select_from_file
    if sampler_func == select_from_file:
        return tuple(f for f in filenames_gen(**selection_kwargs)
                 if include_file(f, band_specs, **selection_kwargs))
    else:
        return tuple(filenames_gen(**selection_kwargs))

