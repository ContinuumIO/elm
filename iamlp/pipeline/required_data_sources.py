from iamlp.acquire.acquire import get_required_data_sources

def required_data_sources_step(args, config, step):
    return get_required_data_sources(args, config, step)
