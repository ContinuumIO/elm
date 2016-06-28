from elm.acquire.acquire import get_download_data_sources

def download_data_sources_step(config, step, executor):
    return get_download_data_sources(config, step, executor)
