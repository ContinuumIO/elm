import os
from sklearn.cluster import MiniBatchKMeans


from iamlp.settings import DOWNLOAD_DIR, delayed, SERIAL_EVAL
from iamlp.selection.filename_selection import _filter_on_filename
from iamlp.readers.hdf4_L2_tools import load_hdf4
from iamlp.readers.local_file_iterators import (get_filenames_for_day,
                                    get_all_filenames_for_product)
from iamlp.selection.geo_selection import (point_in_poly,
                                           points_in_poly,
                                           _filter_band_data)
from iamlp.cli import (add_local_dataset_options,
                       add_ensemble_partial_fit_args)


DEFAULT_BATCH = 4000000
DEFAULT_N_CLUSTERS = 8
DEFAULT_N_ENSEMBLES = 5
DEFAULT_N_SAMPLES_EACH_FIT = 3
DEFAULT_FILES_PER_SAMPLE = 4
def example_init_func(batch_size=1000000, n_clusters=8):
    return MiniBatchKMeans(n_clusters=n_clusters,
                           batch_size=batch_size,
                           compute_labels=True)

def cli():
    parser = ArgumentParser(description="Example kmeans with partial_fit and ensemble model averaging")
    parser.add_argument('output_tag')
    return add_local_dataset_options(add_ensemble_partial_fit_args(parser))

def main():
    args = cli().parse_args()
    if not SERIAL_EVAL:
        from distributed import Executor
        executor = Executor('127.0.0.1:8786')
    if args.band_specs_json:
        with open(args.band_specs_json) as f:
            args.band_specs = json.load(f)
    else:
        args.band_specs = [('long_name', 'Band {} '.format(idx), 'band_{}'.format(idx))
                     for idx in (1, 2, 3, 4, 5, 7, 8, 10, 11)]
    def filenames_gen():
        yield from get_all_filenames_for_product(args.product_number,
                                                 args.product_name,
                                                 pattern='*.hdf')
    selection_kwargs = {}
    models = [example_init_func(n_clusters=args.n_clusters,
                                batch_size=args.batch_size)
              for _ in range(args.n_ensemble)]
    print('Running with args: {}'.format(args))
    models = kmeans_ensemble(   models,
                                args.output_tag,
                                filenames_gen,
                                args.band_specs,
                                n_ensemble=args.n_ensemble,
                                n_samples_to_partial_fit=3,
                                n_per_file=,
                                files_per_sample=4,
                                **selection_kwargs)

    if SERIAL_EVAL:
        return models
    return models.compute()

if __name__ == "__main__":
    models = main()
