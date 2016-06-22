from collections import namedtuple
import pandas as pd
import numpy as np

from iamlp.config import delayed, import_callable
from iamlp.data_selectors.filename_selectors import get_included_filenames
from iamlp.data_selectors.band_selectors import select_from_file

Sample = namedtuple('Sample', 'df band_meta filemeta filename')

def random_image_selection(band_specs, n_rows,
                           **selection_kwargs):

    included_filenames = selection_kwargs['included_filenames']
    filename = np.random.choice(included_filenames)
    df, band_meta, filemeta = select_from_file(filename,
                                               band_specs,
                                               **selection_kwargs)
    if n_rows is not None:
        df = df.iloc[np.random.randint(0, df.shape[0], n_rows)]
    return Sample(df, band_meta, filemeta, filename)

def random_images_selection(sampler_name, sampler_dict, data_sources,
                            **selection_kwargs):
    ds = data_sources[sampler_name]
    band_specs = ds['band_specs']
    dfs, band_metas, filemetas, filenames = [], [], [], []
    if sampler_dict.get('n_rows_per_sample') and sampler_dict.get('files_per_sample'):
        n_rows = sampler_dict['n_rows_per_sample'] // sampler_dict['files_per_sample']
    elif sampler_dict.get('files_per_sample'):
        n_rows = None
    elif sampler_dict.get('n_rows_per_sample'):
        n_rows = sampler_dict['n_rows_per_sample']
    for file_idx in range(sampler_dict['files_per_sample']):
        sample = random_image_selection(band_specs, n_rows,
                                        **selection_kwargs)
        df, band_meta, filemeta, filename = sample[0], sample[1], sample[2], sample[3]
        dfs.append(sample.df)
        band_metas.append(sample.band_meta)
        filemetas.append(sample.filemeta)
        filenames.append(sample.filename)
    return Sample(delayed(pd.concat)(dfs, keys=filenames),
        band_metas,
        filemetas,
        filenames)
