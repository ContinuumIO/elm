__import__('pkg_resources').declare_namespace('elm')
from elm._version import get_versions
__version__ = get_versions()['version']
del get_versions
