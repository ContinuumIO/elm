from iamlp.acquire.acquire import get_required_data_sources

def required_data_sources_step(config, step, executor):
    return get_required_data_sources(config, step, executor)
