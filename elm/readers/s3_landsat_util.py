import logging
import os
import re
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

S3_TIF_DIR = os.environ.get('S3_TIF_DIR')
if not S3_TIF_DIR:
    S3_TIF_DIR = os.path.join(os.path.expanduser('~'), 's3-landsat')

class SceneDownloader:

    def __init__(self, scene_list_gz='scene_list.gz', s3_tif_dir=None):
        self.s3_tif_dir = s3_tif_dir or S3_TIF_DIR
        if not os.path.exists(self.s3_tif_dir):
            os.makedirs(self.s3_tif_dir)
        self.scene_list_gz = scene_list_gz
        if not os.path.exists(scene_list_gz):
            self.download_scene_list()
        self.reload_scene_list()

    def reload_scene_list(self):
        logger.info('Loading LANDSAT scene list from {}'.format(self.scene_list_gz))
        self.df = pd.read_csv(self.scene_list_gz,
                              parse_dates=['acquisitionDate'],
                              compression='gzip')

    def download_scene_list(self):
        scene_list_url = 'http://landsat-pds.s3.amazonaws.com/scene_list.gz'
        if os.path.exists(self.scene_list_gz):
            return
        with urlopen(scene_list_url) as f:
            logger.info('Download {} to {}'.format(scene_list_url, self.scene_list_gz))
            with open(self.scene_list_gz, 'wb') as f2:
                f2.write(f.read())

    def get_scene_list(self, row=33, path=15, max_cloud=10, months=tuple(range(1,13))):
        df = self.df
        sel = df[(df.path == path)&
                 (df.row == row)&
                 (df.cloudCover < max_cloud)&
                 (df.cloudCover != -1.)&
                 (df.acquisitionDate.dt.month.isin(months))]
        return sel

    def lowest_cloud_cover_image(self, **kw):
        sel = self.get_scene_list(**kw)
        return sel[sel.cloudCover == sel.cloudCover.min()]

    def local_file_for_url(self, download_url):
        _, rel_file = download_url.split('.com/')
        full_file = os.path.join(self.s3_tif_dir, rel_file)
        dirr = os.path.dirname(full_file)
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        return full_file


    def get_urls_on_index_page(self, download_url):
        with urlopen(download_url) as f:
            contents = f.read().decode()
        fnames = []
        for ending in ('txt', 'TIF'):
            fnames.extend(set(re.findall('([\w\d_-]+\.{})'.format(ending), contents)))
        assert fnames
        return [download_url.replace('index.html', f) for f in fnames]


    def download_one_file(self, url, fname):
        try:
            print('Download', url, 'to', fname)
            with urlopen(url) as f:
                with open(fname, 'wb') as f2:

                    f2.write(f.read())
        except:
            if os.path.exists(fname):
                os.remove(fname)
            raise


    def download_all_bands(self, download_url):
        urls = self.get_urls_on_index_page(download_url)
        local_files = [self.local_file_for_url(url) for url in urls]
        for url, fname in zip(urls, local_files):
            if not os.path.exists(fname):
                self.download_one_file(url, fname)
        return local_files
