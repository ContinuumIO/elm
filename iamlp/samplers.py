import pandas as pd
import numpy as np

from iamlp.settings import delayed
from iamlp.selection.filename_selection import get_included_files
from iamlp.selection.band_selection import select_from_file

@delayed
def random_image_selection(included_files, band_specs,
                       n_rows=None, **selection_kwargs):
    filename = np.random.choice(included_files)
    df, band_meta, filemeta = select_from_file(filename,
                                               band_specs,
                                               **selection_kwargs)
    if n_rows is not None:
        df = df.iloc[np.random.randint(0, df.shape[0], n_rows)]
    return (df, band_meta, filemeta, filename)

@delayed
def random_image_selection_gen(filenames_gen, n_samples_each_fit, n_per_file,
                   files_per_sample, band_specs, **kwargs):
    included_files = get_included_files(filenames_gen, band_specs, **kwargs)
    for sample_idx in range(n_samples_each_fit):
        dfs = []
        band_metas = []
        filemetas = []
        filenames = []
        rows_added = 0
        for file_idx in range(files_per_sample):
            sample = random_image_selection(included_files, band_specs,
                                        n_rows=n_per_file, **kwargs)
            df, band_meta, filemeta, filename = sample
            dfs.append(df)
            band_metas.append(band_meta)
            filemetas.append(filemeta)
            filenames.append(filename)
        yield (pd.concat(dfs, keys=filenames),
            band_metas,
            filemetas)
