## NASA Contacts

[Dan Duffy, Ph.D.](https://www.nccs.nasa.gov/about_us/leadership/duffy) is our technical contact.  His leadership is in the following projects:

 * [MERRA Analytics Service Project](http://gmao.gsfc.nasa.gov/research/merra/intro.php) .  This is a helpful [brochure](http://gmao.gsfc.nasa.gov/pubs/brochures/MERRA%20Brochure.pdf) about all the data assimilation going into MERRA
 * [Climate Analytics-as-a-Service (CAaaS)](http://www.nas.nasa.gov/SC14/demos/demo29.html)

## Example Data Sets for Testing and Promotion

 * See [Bulk LANDSAT data download here](http://landsat.usgs.gov/Landsat_Search_and_Download.php)
 * [Comprehensive Large-Array Data Stewardship System (CLASS)](http://www.nsof.class.noaa.gov/saa/products/welcome)
 * [Earth Explorer and Bulk Downloader Tools](http://earthexplorer.usgs.gov/) (Create a username and install the bulk downloader tool for your OS [I have only tried Linux])
 * [MODIS 0.5 degree Surface reflectance](https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod09cmg_v006) - [Data access](http://e4ftl01.cr.usgs.gov/MOLT/MOD09CMG.006/)
 * [MODTBGA: MODIS/Terra Thermal Bands Daily L2G Global 1 km SIN Grid V006](https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/modtbga_v006): [HDF files](http://e4ftl01.cr.usgs.gov/MOLT/MODTBGA.006/)
 * [other reflectance datasets that may be easily accessible](https://lpdaac.usgs.gov/dataset_discovery/?f[0]=im_field_product%3A10&f[1]=im_field_data_access%3A60&f[2]=im_field_temporal_range%3A69&f[3]=im_field_temporal_range%3A70&f[4]=im_field_temporal_range%3A71)
 * [TOVS (i.e. vertical high resolution temperature sounds) is an input data set to MERRA](http://disc.sci.gsfc.nasa.gov/legacydata/tovs_5day_readme.html) and we may want to look at it for example projects.
 
Several of the example data sets are HDF4 of HDF-EOS format.  They can be read with `gdal` like this:
```
import gdal
f = '/Users/psteinberg/Downloads/MODTBGA.A2000055.h00v09.006.2015135234839.hdf'
reflect = gdal.Open(f)
datasets = reflect.GetSubDatasets()
arrays = []
for dataset in datasets:
    arrays.append(gdal.Open(dataset[0]).ReadAsArray())
```
 * Many datasets, such as VIIRS, can be downloaded via Anonymous ftp from [ladsweb](https://ladsweb.nascom.nasa.gov/data/). Example:
```bash
ftp
open ladsweb.nascom.nasa.gov
# Enter Anonymous for user and nothing for password
get README
```
The README for ladsweb ftp, retrieved in the snippet above, shows the directory layout of the FTP - it is saved [here in the repo](README_ladsweb.txt)
