from argparse import Namespace
from iamlp.settings import delayed

def get_loader(config, name):
    return config.readers[name]['load'][1]

def get_bounds_func(config, name):
    return config.readers[name]['bounds'][1]

def get_download(config, name):
    return config.downloads[name]

def args_from_spec(ds):
    return Namespace(product_number=ds.product_number, product_name=ds.product_name,
                     years=ds.years, data_days=ds.data_days, band_specs=ds.band_specs)

@delayed
def get_required_data_sources(cli_args, config, step):
    required_data_sources = step['required_data_sources']
    for name in required_data_sources:
        data_source = config.data_sources[name]
        download = get_download(config, data_source['download'])
        args = args_from_spec(data_source)
        download(args=args)
    return True