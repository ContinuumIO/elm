import gc

from matplotlib import pyplot as plt
import gdal
import numpy as np
import xarray as xr



class ElmStore(xr.Dataset):
    # not sure if we want a special variant of the xr.Dataset class or not!

    def __str__(self):
        return "ElmStore:\n" + super().__str__()

