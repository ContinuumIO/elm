from collections import namedtuple
import pandas as pd
import numpy as np

from iamlp.settings import delayed
from iamlp.selection.filename_selection import get_included_filenames
from iamlp.selection.band_selection import select_from_file

Sample = namedtuple('Sample', 'df band_meta filemeta filenames')

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
def random_images_selection(included_filenames, n_samples_each_fit, n_per_file,
                   files_per_sample, band_specs, **kwargs):
    dfs, band_metas, filemetas, filenames = [], [], [], []
    for file_idx in range(files_per_sample):
        sample = random_image_selection(included_filenames, band_specs,
                                    n_rows=n_per_file, **kwargs)
        df, band_meta, filemeta, filename = sample
        dfs.append(df)
        band_metas.append(band_meta)
        filemetas.append(filemeta)
        filenames.append(filename)
    return Sample(pd.concat(dfs, keys=filenames),
        band_metas,
        filemetas,
        filenames)
