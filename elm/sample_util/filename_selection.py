import re

def include_file(*args, **kwargs):
    from elm.sample_util.band_selection import select_from_file
    kwargs['dry_run'] = True
    return select_from_file(*args, **kwargs)

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


def get_args_list(filenames_list, use_file_filter=False, **data_source):
    from elm.sample_util.band_selection import select_from_file
    if use_file_filter:
        return tuple(f for f in filenames_list if include_file(f, **data_source))
    else:
        return tuple(filenames_list)

