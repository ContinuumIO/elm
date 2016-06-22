from collections import namedtuple
import pandas as pd
import numpy as np

from iamlp.config import delayed
from iamlp.selection.filename_selection import get_included_filenames
from iamlp.selection.band_selection import select_from_file

Sample = namedtuple('Sample', 'df band_meta filemeta filename')

@delayed(pure=True)
def random_image_selection(included_filenames, band_specs,
                           n_rows=None, **selection_kwargs):
    filename = np.random.choice(included_filenames)
    df, band_meta, filemeta = select_from_file(filename,
                                               band_specs,
                                               **selection_kwargs)
    if n_rows is not None:
        df = df.iloc[np.random.randint(0, df.shape[0], n_rows)]
    return Sample(df, band_meta, filemeta, filename)

@delayed(pure=True)
def random_images_selection(sampler_name, sampler_dict, data_sources,
                            **selection_kwargs):
    band_specs = data_sources[sampler_name]['band_specs']
    dfs, band_metas, filemetas, filenames = [], [], [], []
    for file_idx in range(sampler_dict['files_per_sample']):
        sample = random_image_selection(kwargs['included_filenames'],
                                        band_specs,
                                        n_rows=sampler['n_per_file'],
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
