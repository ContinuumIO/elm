__import__('pkg_resources').declare_namespace('iamlp')
from iamlp._version import get_versions
__version__ = get_versions()['version']
del get_versions