from argparse import Namespace
from iamlp.config import delayed, import_callable_from_string

def get_loader(config, name):
    return config.readers[name]['load'][1]

def get_bounds_func(config, name):
    return config.readers[name]['bounds'][1]

def get_download(config, name):
    func = config.downloads[name]
    return import_callable_from_string(func, True, func)

def args_from_spec(ds):
    return Namespace(product_number=ds['product_number'],
                     product_name=ds['product_name'],
                     years=ds['years'],
                     data_days=ds['data_days'],
                     band_specs=ds['band_specs'])

def get_download_data_sources(config, step, executor):
    download_data_sources = step['download_data_sources']
    for name in download_data_sources:
        data_source = config.data_sources[name]
        download = get_download(config, data_source['download'])
        args = args_from_spec(data_source)
        download(args=args, config=config)
    return True