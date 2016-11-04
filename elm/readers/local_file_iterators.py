import glob
import logging
import os
import re

logger = logging.getLogger(__name__)


def iter_dirs_of_dirs(**kwargs):
    top_dir = kwargs['top_dir']
    file_pattern = kwargs.get('file_pattern') or None
    logger.debug('Read {} from {}'.format(file_pattern, top_dir))
    for root, dirs, files in os.walk(top_dir):
        if any(os.path.isfile(os.path.join(root, f)) for f in files):
            if (file_pattern and any(re.search(file_pattern, f) for f in files)) or not file_pattern:
                yield root



def iter_files_recursively(**kwargs):
    file_pattern = kwargs.get('file_pattern') or None
    top_dir = kwargs['top_dir']
    logger.debug('Read {} from {}'.format(file_pattern, top_dir))
    if not top_dir or not os.path.exists(top_dir):
        raise ValueError('Expected top_dir ({}) to exist'.format(top_dir))
    for root, dirs, files in os.walk(top_dir):
        if files:
            if file_pattern:
                files = (f for f in files if re.search(file_pattern, f))
            else:
                files = iter(files)
            yield from (os.path.join(root, f) for f in files)

