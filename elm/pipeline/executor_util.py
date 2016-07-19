
def wait_for_futures(futures, executor=None):
    '''Abstraction of waiting for mapped results
    that works for any type of executor or no executor'''
    if not executor:
        results = list(futures)
    elif hasattr(executor, 'gather'): # distributed
        from distributed import progress
        progress(futures)
        results = executor.gather(futures)
    else:
        results = []
        for fut in as_completed(futures):
            if fut.exception():
                raise ValueError(fut.exception())
            results.append(fut.result())
    return results

def no_executor_submit(func, *args, **kwargs):
    return func(*args, **kwargs)