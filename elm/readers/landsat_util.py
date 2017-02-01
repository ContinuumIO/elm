'''
This module is helpful for parsing LANDSAT MTL files.

It is an adaptation from:
    NASA - National Program Python package for use with NASA data and GIS
    https://github.com/NASA-DEVELOP/dnppy/blob/master/dnppy/landsat/landsat_metadata.py

An example MTL file:
    http://landsat-pds.s3.amazonaws.com/L8/015/033/LC80150332013207LGN00/LC80150332013207LGN00_MTL.txt

Can be parsed with:
    landsat_metadata(filename)
'''

from datetime import datetime

class landsat_metadata:
    """
    A landsat metadata object. This class builds is attributes
    from the names of each tag in the xml formatted .MTL files that
    come with landsat data. So, any tag that appears in the MTL file
    will populate as an attribute of landsat_metadata.
    """

    def __init__(self, filename):
        """
        Parse MTL file (filename)
        """

        # custom attribute additions
        self.FILEPATH           = filename
        self.DATETIME_OBJ       = None

        # product metadata attributes
        self.LANDSAT_SCENE_ID   = None
        self.DATA_TYPE          = None
        self.ELEVATION_SOURCE   = None
        self.OUTPUT_FORMAT      = None
        self.SPACECRAFT_ID      = None
        self.SENSOR_ID          = None
        self.WRS_PATH           = None
        self.WRS_ROW            = None
        self.NADIR_OFFNADIR     = None
        self.TARGET_WRS_PATH    = None
        self.TARGET_WRS_ROW     = None
        self.DATE_ACQUIRED      = None
        self.SCENE_CENTER_TIME  = None

        # image attributes
        self.CLOUD_COVER        = None
        self.IMAGE_QUALITY_OLI  = None
        self.IMAGE_QUALITY_TIRS = None
        self.ROLL_ANGLE         = None
        self.SUN_AZIMUTH        = None
        self.SUN_ELEVATION      = None
        self.EARTH_SUN_DISTANCE = None    # calculated for Landsats before 8.

        # read the file and populate the MTL attributes
        self._read(filename)

    def _read(self, filename):
        """ reads the contents of an MTL file """


        fields = []
        values = []

        with open(filename, 'r') as metafile:
            metadata = metafile.readlines()

        for line in metadata:
            # skips lines that contain "bad flags" denoting useless data AND lines
            # greater than 1000 characters. 1000 character limit works around an odd LC5
            # issue where the metadata has 40,000+ characters of whitespace
            bad_flags = ["END", "GROUP"]
            if not any(x in line for x in bad_flags) and len(line) <= 1000:
                try:
                    line = line.replace("  ", "")
                    line = line.replace("\n", "")
                    field_name, field_value = line.split(' = ')
                    fields.append(field_name)
                    values.append(field_value)
                except:
                    pass

        for i in range(len(fields)):

            # format fields without quotes,dates, or times in them as floats
            if not any(['"' in values[i], 'DATE' in fields[i], 'TIME' in fields[i]]):
                setattr(self, fields[i], float(values[i]))
            else:
                values[i] = values[i].replace('"', '')
                setattr(self, fields[i], values[i])

        # create datetime_obj attribute (drop decimal seconds)
        dto_string          = self.DATE_ACQUIRED + self.SCENE_CENTER_TIME
        self.DATETIME_OBJ   = datetime.strptime(dto_string.split(".")[0], "%Y-%m-%d%H:%M:%S")

