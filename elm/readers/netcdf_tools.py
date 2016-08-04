import netCDF4

def load_netcdf_variable(fpath, variable):
    """Open the given netcdf file and return the variable as a numpy array
    """
    with netcdf4.Dataset(fpath, 'r') as f:
        return f[variable][:]

def netcdf_attrs(fpath, attr=None):
    """Return a dictionary of attribute name -> attribute value
       If attr is given, return a dictionary of that attribute only.
       If that attribute doesn't exist, return {attr: None}
    """
    with netcdf4.Dataset(fpath, 'r') as f:
        if attr is not None:
            try:
                res = f.getncattr(attr)
                return {attr: res}
            except AttributeError:
                return {attr: None}
            
        attrs = {ncattr: f.getncattr(ncattr) for ncattr in f.ncattrs()}
            
        for var in f.variables:
            attrs[var] = {ncattr: f[var].getncattr(ncattr) for ncattr in f[var].ncattrs()}

        for group in f.groups:
            attrs[group] = {ncattr: f[group].getncattr(ncattr) for ncattr in f[group].ncattrs()}
            
    return attrs

def netcdf_variables(fpath):
    """Return a dict of all of the variables with their attribute/value pairs for the given file
    """
    with netcdf4.Dataset(fpath, 'r') as f:
        d = {}
        for varname in f.variables:
            d[varname] = {attr: f[varname].getncattr(attr) for attr in f[varname].ncattrs()}
        return d
