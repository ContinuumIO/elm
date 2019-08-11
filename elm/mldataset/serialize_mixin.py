from __future__ import (absolute_import, division, print_function,)
import dill

class SerializeMixin:
    '''A mixin for serialization of estimators via dill'''
    def dumps(self, protocol=None, byref=None, fmode=None, recurse=None):
        '''pickle (dill) an object to a string
        '''
        getattr(self, '_close', lambda: [])()
        return dill.dumps(self, protocol=protocol,
                          byref=byref, fmode=fmode, recurse=recurse)

    def dump(self, file, protocol=None, byref=None, fmode=None, recurse=None):
        '''pickle (dill) an object to a file'''
        getattr(self, '_close', lambda: [])()
        return dill.dump(self, file, protocol=protocol,
                         byref=byref, fmode=fmode, recurse=recurse)
