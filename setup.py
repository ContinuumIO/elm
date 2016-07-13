import os
import glob
from setuptools import setup

import versioneer

version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()

setup(name='elm',
      version=version,
      cmdclass=cmdclass,
      description='Ensemble Learning Models',
      install_requires=['deap',
                       'graphviz',
                       'pysptools',],
      packages=['elm',
                'elm.acquire',
                'elm.config',
                'elm.model_selection',
                'elm.pipeline',
                'elm.preproc',
                'elm.readers',
                'elm.sample_util',
                'elm.scripts',
                'elm.writers',
                ],
      data_files=[('elm', glob.glob(os.path.join('elm', 'examples', '*.yaml'))),
                  ('elm.acquire',
                   glob.glob(os.path.join('elm', 'acquire', 'metadata', '*'))),
                  ('elm.config',
                   glob.glob(os.path.join('elm', 'config', 'defaults', '*.yaml')))
                 ],
      entry_points={
        'console_scripts': [
            'elm-download-ladsweb = elm.acquire.ladsweb_ftp:main',
            'elm-main = elm.scripts.main:main',
            'elm-collect-ladsweb-metadata = elm.acquire.ladsweb_ftp:get_sample_main',

        ]},)