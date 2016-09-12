from argparse import Namespace
import calendar
import datetime
from elm.config import  import_callable

def get_download(config, name):
    func = config.downloads[name]
    return import_callable(func, True, func)

def args_from_spec(ds):
    data_days = ds['data_days']
    if data_days in ('all', ['all']):
        data_days = get_default_data_days(ds['years'])
    return Namespace(product_number=ds['product_number'],
                     product_name=ds['product_name'],
                     years=ds['years'],
                     data_days=data_days,
                     band_specs=ds['band_specs'])

def get_default_data_days(yr):
    today = datetime.datetime.utcnow()
    if yr == today.year:
        today_data_day = 0
        for mo in range(1, today.month):
            eom = calendar.monthrange(yr, mo)[-1]
            today_data_day += eom
        today_data_day += today.day
        data_days = list(range(1, today_data_day + 1))
    else:
        data_days = list(range(1, 366))
    return data_days

def get_download_data_sources(config, step, client):
    download_data_sources = step['download_data_sources']
    for name in download_data_sources:
        data_source = config.data_sources[name]
        download = get_download(config, data_source['download'])
        args = args_from_spec(data_source)
        download(args=args, config=config)
    return True