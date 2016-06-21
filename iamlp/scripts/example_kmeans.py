
from argparse import ArgumentParser
import datetime
from functools import partial
import os
from sklearn.cluster import MiniBatchKMeans
from dask.diagnostics import ProgressBar

from iamlp.settings import DOWNLOAD_DIR, delayed, SERIAL_EVAL, DASK_EXECUTOR
from iamlp.selection.filename_selection import _filter_on_filename
from iamlp.readers.hdf4_L2_tools import load_hdf4
from iamlp.readers.local_file_iterators import (get_filenames_for_day,
                                    get_all_filenames_for_product)
from iamlp.selection.geo_selection import (point_in_poly,
                                           points_in_poly,
                                           _filter_band_data)
from iamlp.cli import (add_local_dataset_options,
                       add_ensemble_partial_fit_args)

from iamlp.ensemble.ensemble_base import ensemble
from iamlp.writers.serialize import serialize
from iamlp.partial_fit import partial_fit
from iamlp.model_averaging.kmeans import (_kmeans_add_within_class_var,
                                          kmeans_model_averaging)
from iamlp.settings import executor_context
def example_init_func(batch_size=1000000, n_clusters=8):
    return MiniBatchKMeans(n_clusters=n_clusters,
                           batch_size=batch_size,
                           compute_labels=True)

def cli():
    parser = ArgumentParser(description="Example kmeans with partial_fit and ensemble model averaging")
    parser.add_argument('output_tag')
    return add_local_dataset_options(add_ensemble_partial_fit_args(parser))


def main():
    started = datetime.datetime.now()
    args = cli().parse_args()
    args.years = args.years or [2016]
    args.data_days = args.data_days or list(range(1, 70))
    if args.band_specs_json:
        with open(args.band_specs_json) as f:
            args.band_specs = json.load(f)
    else:
        args.band_specs = [('long_name', 'Band {} '.format(idx), 'band_{}'.format(idx))
                     for idx in (1, 2, 3, 4, 5, 7, 8, 10, 11)]
    included_filenames = tuple(get_all_filenames_for_product(args.product_number,
                                                 args.product_name,
                                                 pattern='*.hdf'))
    selection_kwargs = {}
    args.n_per_file = args.batch_size // args.files_per_sample
    print('Running with args: {}'.format(args))
    NO_SHUFFLE = 1
    def init_models(models=None):
        return [example_init_func(n_clusters=args.n_clusters,
                                  batch_size=args.batch_size)
                for _ in range(args.n_models)]
    partial_fit_kwargs = {
        'included_filenames': included_filenames,
        'n_samples_each_fit': args.n_samples_each_fit,
        'n_per_file': args.n_per_file,
        'files_per_sample': args.files_per_sample,
        'post_fit_func': partial(_kmeans_add_within_class_var, args.n_clusters),
        'selection_kwargs': selection_kwargs,
        'band_specs': args.band_specs,
    }
    ensemble_kwargs = {
        'no_shuffle': args.no_shuffle,
        'n_generations': args.n_generations,
        'partial_fit_kwargs': partial_fit_kwargs,
    }
    with executor_context() as (executor, get_func):
        with ProgressBar() as progress:
            ensemble_kwargs['get_func'] = get_func
            models = ensemble(init_models,
                             args.output_tag,
                             partial(kmeans_model_averaging,
                                     args.n_clusters,
                                     {'n_clusters': args.n_clusters,}),
                             **ensemble_kwargs)
            if DASK_EXECUTOR == 'DISTRIBUTED':
                get = get_func
            else:
                get = None
                #print('keys', [tuple(m.dask.keys()) for m in models])
            if not SERIAL_EVAL:
                m = models.compute(get=get_func)
                print('m', m)
                models = [m.compute(get=get_func) for m in models.compute(get=get_func)]
            for model_idx, model in enumerate(models):
                serialize(args.output_tag + '_{}'.format(model_idx), model)
            ended = datetime.datetime.now()
            print('Ran from', started, 'to', ended, '({} seconds)'.format((ended - started).total_seconds()))
        return models


if __name__ == "__main__":
    models = main()
