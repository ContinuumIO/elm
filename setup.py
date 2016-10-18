import os
import glob
from setuptools import setup, find_packages

import versioneer

version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()
yamls = glob.glob(os.path.join('elm', 'config', 'defaults', '*'))
data_files = [('elm', yamls)]
setup(name='elm',
      version=version,
      cmdclass=cmdclass,
      description='Ensemble Learning Models',
      include_package_data=True,
      install_requires=[],
      packages=find_packages(),
      data_files=data_files,
      entry_points={
        'console_scripts': [
            'elm-main = elm.scripts.main:main',
            'elm-run-all-tests = elm.scripts.run_all_tests:run_all_tests',
        ]})
