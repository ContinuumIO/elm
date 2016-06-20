import os
import glob
from setuptools import setup

import versioneer

version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()

setup(name='iamlp',
      version=version,
      cmdclass=cmdclass,
      description='Image Analysis Machine Learning Pipeline',
      packages=['iamlp',
                'iamlp.acquire',
                'iamlp.readers',
                'iamlp.writers',
                'iamlp.scripts',
                'iamlp.selection',
                'iamlp.partial_fit',
                'iamlp.model_averaging',
                'iamlp.ensemble'],
      data_files=[],
      entry_points={
        'console_scripts': [
            'iamlp-download-ladsweb = iamlp.acquire.ladsweb_ftp:main',
            'iamlp-kmeans-example = iamlp.scripts.example_kmeans:main'

        ]},)