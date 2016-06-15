*************************************
*LAADS and DATAPOOL DATA ReadMe File*
*************************************

The Level 1 and Atmosphere Archive and Distribution System (LAADS) provides 
land and Atmosphere science communities and general users with rapid and 
flexible access to large volumes of MODIS data from the NASA EOS Terra and 
Aqua spacecraft, MAS data from NASA aircraft campaigns, VIIRS data from 
NASA/NOAA Suomi NPP Spacecraft, and MERIS data from ESA's ENVISAT satellite.

The LAADS system takes advantage of low cost of disk storage to retain several 
petabytes of data on-line. The LAADS now archives MODIS Level 0 data acquired 
from EDOS, MODIS Atmosphere and Land Level 1, Level 2, Level 3,  and Level 4 
products produced at MODAPS,  NPP Level 0 (RDR) from IDPS, Level 1 (SDR),  
Level 2 and Level 3 (EDR) products produced at Land PEATE,  MERIS Level 1 
(FR and RR) Products,  eMAS data from  various campaigns collected over the 
past two decades,  and ancillary data acquired from GES DAAC and  other 
sources. Following sections explain LAADS data archive and distribution 
process by instrument.

The users not only can browse, select, and order data through LAADS website, 
but also can directly access LAADS archive by using LAADS FTP server, 
ftp://ladsftp.nascom.nasa.gov, without having to submit orders.  Users can 
also search and order data by using LAADS Web Services 
(http://ladsweb.nascom.nasa.gov/data/web_services.html).

LAADS is designed to perform searches and orders in seconds except in 
circumstances requiring post processing or regeneration of data through 
Processing On Demand (POD). LAADS includes a wide variety of post processing 
capabilities included in response to request from the user community. These 
include metadata searches, recurring temporal searches, geographic 
sub-setting, parameter sub-setting, sub-sampling, masking, mosaicing, 
reprojection, and GeoTIFF parameter conversion.

*******
*MODIS*
*******

MODIS (Moderate Resolution Imaging Spectroradiometer) is a key instrument 
aboard the Terra (EOS AM) and Aqua (EOS PM) satellites. Terra's orbit around 
the Earth is timed so that it passes from north to south across the equator in 
the morning, while Aqua passes south to north over the equator in the 
afternoon. Terra MODIS and Aqua MODIS are viewing the entire Earth's surface 
every 1 to 2 days, acquiring data in 36 spectral bands. These data will 
improve our understanding of global dynamics and processes occurring on the 
land, in the oceans, and in the lower atmosphere. MODIS is playing a vital 
role in the development of validated, global, interactive Earth system models 
able to predict global change accurately enough to assist policy makers in 
making sound decisions concerning the protection of our environment.

At present, LAADS MODIS archive consists of data products from Collection 3 to 
Collection 6 and product Level 0 to Level 4. The range of data available for 
distribution under each Collection is from a few products for certain years to 
all products for the whole mission as follows:

- MODIS Collection 3 - Atmosphere 2002 Golden Data Set
- MODIS Collection 4 - Atmosphere 2002 Golden Data set
- MODIS Collection 4 - Gap Filled and Smoothed Land Products for NACP
- MODIS Collection 4.1 - Land Surface Temperature
- MODIS Collection 5 - L1, Atmosphere (Aqua 2002-2008, Terra 2000-March 2010) 
  and Land 
- MODIS Collection 5.1 - Selected Atmosphere and Land Products
- MODIS Collection 5.5 - Selected Land Products
- MODIS Collection 6 - L1. Atmosphere and Land data will be available after 
  2012

MODIS products made available to users through LAADS system are grouped by 
satellite (Terra, Aqua, or combined) and by processing level in the following 
manner:

- Terra & Aqua Level 0 Products
- Terra & Aqua Level 1 Products
- Terra & Aqua Atmosphere Level 2 Products
- Terra & Aqua/Combined Atmosphere Level 3 Products
- Terra & Aqua Land Level 2 Products
- Terra & Aqua/Combined Land Level 3/Level 4 CMG Products
- Terra & Aqua Land Level 3/Level 4 Daily Tiled LST Products
- Combined Land Level 3/Level 4 4-Day Tiled Products
- Terra & Aqua/Combined Land Level 3/Level 4 8-Day Tiled Products
- Terra & Aqua/Combined Land Level 3/Level 4 16-Day Tiled Products
- Terra & Aqua/Combined Land Level 3/Level 4 Monthly Tiled Products
- Terra & Aqua/Combined Land Level 3/Level 4 Yearly Tiled Products
- Terra NACP 8-Day and Annual Tiled Products

To facilitate MODIS Level 2 swath products users in  delineating data 
boundaries, LAADS has developed a simple text, GeoMeta Product file. This file,
taken directly from the Collection 5 MODIS Geolocation Products, MOD03/MYD03, 
contains granule corner-point latitudes and longitudes along with 
north/south/east/west bounding coordinates. Also, there are a number of other 
parameters such as the Collection, OrbitNumber, and DayNightFlag included in 
this file. For a complete information access the GeoMeta readme file at:

  ftp://ladsweb.nascom.nasa.gov/geoMeta/README


There are 2 types of data archives on LAADS, a permanent archive that is 
available forever and a rolling archive. The rolling archive, such as Datapool,
saves products on rolling basis for at least 30 days (may be longer depending 
on the disk usage) before data go offline. MODIS data, either LAADS permanent 
archive data or rolling-archive data in the Datapool, are stored in LAADS in a 
universal and predefined directory structure based on ArchiveSet (AS) 
(associated with Collections),  ESDT,  Data-Year,  and Data-Day:

/allData/Archiveset number (AS)/ESDT/YYYY/DDD/  where ESDT is the product 
shortname, YYYY is the 4-digit data year, and DDD is 3-digit data day number. 
Please note that the product shortname for Terra MODIS start with MOD and the 
same with Aqua MODIS start with MYD.

For example,   the Collection 5 Aqua Geolocation data (MYD03) of data-day 
2006042,  is stored in archive  directory "allData/5/MYD03/2006/042/".

A brief description of MODIS data available for download from the LAADS ftp 
server  listed by ArchiveSet is given below:

AS3 - Contains MODIS Atmosphere "Golden-Month" products from Collection 3 
reprocessing  performed in 2002.Terra and Aqua September 2002 "golden month" 
samples are archived for historical reference.

AS4 - MODIS Atmosphere "Golden-Month" products from Collection 4 reprocessing 
completed in 2003. Terra and Aqua September 2002 "golden month" samples are 
archived for historical reference.  Ancillary products are also archived.

AS404 - This archive set contains smooth, gap-filled LAI and EVI/NDVI products 
generated under the North American Carbon Program (NCAP) using MODIS 
Collection 4 Land data as input. This archive also includes input collection 4 
Land surface reflectance data and output LAI and NDVI data sets.

AS41 - Contains C4.1 MODIS Land Surface Temperature (LST) suite of products 
from C4.1 reprocessing that used the Collection 4 LST algorithm with the 
Collection 5 L1B and Cloud Mask data as input. This reprocessing addressed 
some of the issues identified in the Collection 5 LST algorithm such as 
underestimation of LST over desert and other arid areas, cloud contamination 
of LST in the L2 product.

AS5 - Collection 5 MODIS data - L1, Atmosphere and Land Products from 
Collection 5 reprocessing of Aqua and Terra MODIS. This reprocessing includes 
C5 calibration changes to the MODIS Terra and Aqua and all of the C5 changes 
to the MODIS Land data processing algorithms. Please note that C5 Atmosphere 
data is replaced by C5.1 data.

AS51 - Collection 5.1 Atmosphere data and image files. This archive set also 
contains  Annual Land Cover products (MCD12Q1 and MCD12C1) from the C51 
reprocessing of the product using the C51 changes delivered by the science 
team to address issues identified in the C5 Land Cover products.

AS55 - Collection 5.5 Land data - This contains improved version of the 
GPP/PSNnet product generated by the Science Team at the Montana SCF. Product 
is cleaned for cloud contamination and is gap filled.

AS6 - Collection 6 MODIS data - L1, Atmosphere and Land Products from 
Collection 6 reprocessing of Aqua and Terra. This reprocessing includes C6 
calibration changes to the MODIS Terra and Aqua and all of the C6 changes to 
the MODIS Land data processing algorithms. So far, only C6 L1 data is 
available. C6 Atmosphere and Land data will be available in 2013 and 2014 
respectively.

AS1003 - North American Carbon Program (NCAP) smooth,  gap-fill LAI,  EVI and 
NDVI 8-day data sets at 250m and 500m resolution for VI,  and 1km for LAI. 
This data set uses Collection 5 Land data as input.

MAIAC MODIS Data for Amazon Basin:
 
 The MultiAngle Implementation of Atmospheric Correction (MAIAC) is a new 
advanced algorithm which uses time series analysis and a combination of pixel- 
and image-based processing to improve accuracy of cloud detection, aerosol 
retrievals and atmospheric correction. Current dataset presents the full 
geophysical data record over Amazon basin (~55 600km tiles) from both MODIS 
Terra (from 2000) and Aqua (from 2002). MAIAC provides suites of 1km 
atmospheric and surface gridded products which include cloud mask, column water 
vapor and aerosol optical depth, type (background, biomass burning or dust), 
and Angstrom parameter, and surface spectral BRF, albedo and Ross-Thick 
Li-Sparse (RTLS) BRDF model parameters in 7 land and 5 unsaturated ocean bands. 
The 500m gridded land BRF is also produced for MODIS bands 1-7. The BRDF is 
currently reported every 8 days. MAIAC data consist of the following three 
products:
 
 1. Daily Surface Reflectance  (MAIACBRF)
 2. Daily Aerosol Optical Thickness (MAIACAOT)
 3. 8-day BRDF model parameters (MAIACRTLS)
 
 
 For more information, including the data specification and references, please 
see MAIAC_AmazonRelease.docx found at:
 
 ftp://ladsweb.nascom.nasa.gov/MAIAC/
 
 
 The file naming convention is as follows:
 
 MAIACXXX.hHHvVV.YYYYDDDHHMM.hdf
 
 where XXX is the data file type (BRF, AOT, RTLS), HH and VV are the local 
 tile number, YYYYDDDHHMM is the year, Julian day, hour and minute of the 
 corresponding 5-minute MODIS L1B granule. 
 
 The directory structure is:
 
 MAIAC/(Aqua|Terra)/MAIACXXX/hHHvVV/YYYY/
 
 where XXX is the data file type (BRF, AOT, RTLS), HH and VV are the local 
 tile number, YYYY is the year.

*******
*VIIRS*
*******

Visible Infrared Imaging Radiometer Suite (VIIRS) is a scanning radiometer on 
board the NPP satellite. The VIIRS collects visible and infrared imagery and 
radiometric measurements of the land, atmosphere, cryosphere, and oceans. It 
extends and improves upon a series of measurements initiated by the Advanced 
Very High Resolution Radiometer (AVHRR) and the Moderate Resolution Imaging 
Spectroradiometer (MODIS). VIIRS data is used to measure cloud and aerosol 
properties, ocean color, sea and land surface temperature, ice motion and 
temperature, fires, and Earth's albedo. Climatologists use VIIRS data to 
improve our understanding of global climate change.

Like MODIS, VIIRS is a multi-disciplinary sensor providing data for the ocean, 
land, aerosol, and cloud research and operational users. VIIRS spectral 
coverage will allow for data products similar to those from SeaWiFS as well as 
SST, a standard MODIS product.

LAADS provide a number of VIIRS data products to the users. These products 
include:

- NPP Level 0 Products
- NPP Level 1 Products
- NPP Level 1 5-Minute Products
- NPP Level 1 Daily Products
- NPP Level 2 Products
- NPP Level 2 5-Minute Products
- NPP Level 2G Daily Products
- NPP Level 3 Products
- NPP Level 3 Daily Products
- NPP Level 3 Daily Tiled Products
- NPP Level 3 8-Day Tiled Products
- NPP Level 3 16-Day Tiled Products
- NPP Level 3 17-Day Tiled Products
- NPP Level 3 Monthly Tiled Products
- NPP Level 3 Quarterly Tiled Products

As in the case of MODIS data, VIIR data are also stored in LAADS in a 
predefined directory structure based on ArchiveSet (AS), ESDT, Data-Year, and 
Data-Day:

/allData/ArchiveSet/ESDT/YYYY/DDD/.

A brief description of Suomi-NPP VIIRS data available on LAADS ftp server for 
user download is given below listed by ArchiveSet:

AS3000 - NPP data - An archive of NPP VIIRS xDRs generated at IDPS. This 
contains VIIRS xDRs in native IDPS granule size (86 secs) in HDF5 format and 
these granules aggregated at Land PEATE to nearest 5min size in HDF4 format.

AS3001 - NPP data - An archive of NPP VIIRS xDRs generated at Land PEATE by 
processing the RDRs using the IDPS OPS (Operational) code. All xDR granules 
are of ~5min size and are produced in HDF4 format. Also, this archive contains 
L2G and L3 daily and multi-day DDRs (Diagnostic Data Records) generated at 
Land PEATE by running the MODIS algorithms ported to use the VIIRS L2 xDR as 
input  produced at Land PEATE.

AS3002 - NPP data - An archive of NPP VIIRS xDRs generated at Land PEATE by 
processing the RDRs, using the Land PEATE Adjusted version of the IDPS OPS 
codes and new algorithms delivered by the Science Teams. All xDR granules are 
of ~5min size and are produced in HDF4 format. Also contains L2G and L3 daily 
and multi-day DDRs generated at Land PEATE by running the MODIS algorithm 
using the VIIRS xDR data generated at Land PEATE as input.

AS 3110 -- NPP data – An archive the C1.1 reprocessing data, the follow on to 
the C1 reprocessing.  For this data, the best variations of the OPS L1, L2 
and limited L3 codes were in use as well as the LPEATE Science DDRs.  The 
LUTs in use by the L1 codes were those provided by the VCST group specifically 
for the reprocessing, so were not the same as those in use for the C1 
reprocessing.  The reprocessing data period started 1/19/12, same as AS3100 
and  currently anticipate a continual reprocessing of the latest data month 
with revised LUTs from VCST.  No end time for this has been planned.
 
AS 3144 -- NPP data – A partner archive set to AS3110.  After a significant 
amount of reprocessing was completed in AS3110, LPEATE realized that the 
stray light correction was not working for the NPP_VDNE_L1 products.  The 
integration of the code was corrected for the SDR code (PGE302b), and at the 
request of the science team, scaling implemented for the NPP_VDNE_L1 product.  
In order to maintain a consistent non-scaled dataset in AS3110, but still 
accommodate the needs of the science team, a full reprocessing of the 
NPP_VDNE_L1 products using the same LUTs as the C1.1 was executed in AS3144.  
Only the NPP_VDNE_L1 products are retained in LAADS.

******
*eMAS*
******

The enhanced MODIS Airborne Simulator (eMAS) is an airborne scanning 
spectrometer that acquires high spatial resolution imagery of cloud and 
surface features from its vantage point on-board a NASA ER-2 high-altitude 
research aircraft. Data acquired by the eMAS are used to define, develop, test,
and refine algorithms for the Moderate Resolution Imaging Spectroradiometer 
(MODIS). 

The enhanced MODIS Airborne Simulator (eMAS) data is now available through our 
anonymous FTP site on ladsweb.nascom.nasa.gov. The directory structure is as 
follows:

/MAS/<flight number>

The following table provides information about eMAS campaigns and corresponding
dates and flight numbers. These flight numbers can be used to locate data from 
the campaign on LAADS.

+----------------+-------------+----------------------------------------------+
| Campaign       | Dates       | Flight Numbers                               |
+-----------------+-------------+---------------------------------------------+
| TC4            | 07/07/2007- | 07-915, 07-916, 07-918, 07-919, 07-920,      |
|                | 08/08/207   | 07-921                                       |
+----------------+-------------+----------------------------------------------+
| CLASIC         | 06/07/2007- | 07-619, 07-621, 07-622, 07-625, 07-626,      |
|                | 06/30/2007  | 07-627, 07-628, 07-630, 07-631, 07-632       |
+----------------+-------------+----------------------------------------------+
| CCVEX          | 07/24/2006- | 06-610, 06-611, 06-612, 06-613, 06-614,      |
|                | 08/15/2006  | 06-615, 06-616, 06-617, 06-618, 06-619,      |
|                |             | 06-620, 06-621, 06-622                       |
+----------------+-------------+----------------------------------------------+
| DFRC 05        | 10/19/2005- | 06-901,  06-907,  06-908                     |
|                | 12/09/2005  |                                              |
+----------------+-------------+----------------------------------------------+
| DFRC 06        | 09/26/2006- | 06-602,  06-603,  06-630                     |
|                | 10/13/2006  |                                              |
+----------------+-------------+----------------------------------------------+
| TCSP           | 07/01/2005- | 05-921, 05-922, 05-923, 05-924, 05-925,      |
|                | 07/28/2005  | 05-926, 05-927, 05-928, 05-929, 05-930,      |
|                |             | 05-931, 05-932, 05-933, 05-934, 05-935,      |
|                |             | 05-936                                       |
+----------------+-------------+----------------------------------------------+
| SSMIS #3       | 03/07/2005- | 05-910, 05-911, 05-912                       |
|                | 03/16/2005  |                                              |
+----------------+-------------+----------------------------------------------+
| SSMIS #2       | 12/02/2004- | 05-904, 05-905                               |
|                | 12/20/2004  |                                              |
+----------------+-------------+----------------------------------------------+
| SSMIS #1       | 03/15/2004- | 04-912, 04-913, 04-914, 04-917, 04-918       |
|                | 03/26/2004  |                                              |
+----------------+-------------+----------------------------------------------+
| DFRC 04        | 02/11/2004  | 04-906, 04-907, 04-919, 04-920, 04-921,      |
|                | 10/28/2004  | 04-940, 04-941, 04-942, 04-943, 04-944,      |
|                |             | 04-945, 04-946, 04-953, 04-954, 04-955,      |
|                |             | 04-956, 04-959, 05-901, 05-902               |
+----------------+-------------+----------------------------------------------+
| ATOST          | 11/17/2003- | 04-615, 04-616, 04-619, 04-621, 04-622,      |
|                | 12/17/2003  | 04-623, 04-624, 04-625, 04-626, 04-627       |
+----------------+-------------+----------------------------------------------+
| GLAS           | 10/16/2003- | 04-605, 04-606, 04-607                       |
|                | 10/18/2003  |                                              |
+----------------+-------------+----------------------------------------------+
| DFRC 03        | 04/09/2003- | 03-623, 03-624, 03-942, 03-943, 03-944,      |
|                | 06/29/2003  | 03-945, 03-946, 03-947                       |
+----------------+-------------+----------------------------------------------+
| THORPEX        | 02/18/2003- | 03-610, 03-611, 03-612, 03-613, 03-614,      |
|                | 04/07/2003  | 03-615, 03-616, 03-617, 03-618, 03-619,      |
|                |             | 03-622, 03-625, 03-931, 03-932, 03-933,      |
|                |             | 03-934, 03-935                               |
+----------------+-------------+----------------------------------------------+
| TX-2002        | 11/20/2002- | 03-911, 03-912, 03-913, 03-914, 03-915,      |
|                | 12/13/2002  | 03-916, 03-917, 03-918, 03-919, 03-920,      |
|                |             | 03-921, 03-922, 03-923                       |
+----------------+-------------+----------------------------------------------+
| DFRC 02        | 08/07/2002- | 02-926, 02-927, 02-928, 02-929, 02-930,      |
|                | 08/10/2002  | 02-931, 02-932, 02-959, 02-960, 02-961       |
+----------------+-------------+----------------------------------------------+
| CRYSTAL-FACE   | 07/01/2002- | 02-941, 02-942, 02-943, 02-944, 02-945,      |
|                | 07/31/2002  | 02-946, 02-947, 02-948, 02-949, 02-950,      |
|                |             | 02-951, 02-952, 02-953, 02-954, 02-955,      |
|                |             | 02-956, 02-957                               |
+----------------+-------------+----------------------------------------------+
| CAMEX 4        | 08/13/2001- | 01-122, 01-130, 01-131, 01-132, 01-133,      |
|                | 09/26/2001  | 01-135, 01-136, 01-137, 01-138, 01-139,      |
|                |             | 01-140, 01-141, 01-142, 01-143               |
+----------------+-------------+----------------------------------------------+
| DFRC 01        | 03/08/2001- | 01-046, 01-047, 01-048, 01-059, 01-061,      |
|                | 10/03/2001  | 01-062, 01-093, 01-099, 02-602, 02-603       |
+----------------+-------------+----------------------------------------------+
| CLAMS          | 07/09/2001- | 01-100, 01-101, 01-102, 01-103, 01-104,      |
|                | 08/03/2001  | 01-105, 01-106, 01-107, 01-108, 01-109,      |
|                |             | 01-110                                       |
+----------------+-------------+----------------------------------------------+
| TX-2001        | 03/14/2001- | 01-049, 01-050, 01-051, 01-052, 01-053,      |
|                | 04/05/2001  | 01-054, 01-055, 01-056, 01-057, 01-058       |
+----------------+-------------+----------------------------------------------+
| Pre-SAFARI     | 07/25/2000- | 00-137, 00-140, 00-142                       |
|                | 08/03/2000  |                                              |
+----------------+-------------+----------------------------------------------+
| SAFARI 2000    | 08/06/2000- | 00-143, 00-147, 00-148, 00-149, 00-150,      |
|                | 09/25/2000  | 00-151, 00-152, 00-153, 00-155, 00-156,      |
|                |             | 00-157, 00-158, 00-160, 00-175, 00-176,      |
|                |             | 00-177, 00-178, 00-179, 00-180               |
+----------------+-------------+----------------------------------------------+
| WISC-T2000     | 02/24/2000- | 00-062, 00-063, 00-064, 00-065, 00-066,      |
|                | 03/13/2000  | 00-067, 00-068, 00-069, 00-070, 00-071       |
+----------------+-------------+----------------------------------------------+
| Wallops 2000   | 05/24/2000- | 00-110, 00-111                               |
|                | 05/25/2000  |                                              |
+----------------+-------------+----------------------------------------------+
| DFRC 00        | 02/03/2000- | 00-057, 00-058, 00-059, 00-060, 00-112,      |
|                | 10/13/2000  | 00-113, 00-114, 00-115, 00-116, 00-117,      |
|                |             | 00-118, 00-119, 01-001, 01-003               |
+----------------+-------------+----------------------------------------------+
| Hawaii 2000    | 04/04/2000- | 00-077, 00-079, 00-080, 00-081, 00-082,      |
|                | 04/27/2000  | 00-083, 00-084, 00-086, 00-087, 00-088,      |
|                |             | 00-089, 00-090, 00-091, 00-092, 00-093       |
+----------------+-------------+----------------------------------------------+
| Patrick        | 05/07/1999- | 99-065, 99-067, 99-068, 99-069, 99-070,      |
|                | 05/27/1999  | 99-071, 99-072, 99-073, 99-074, 99-075,      |
|                |             | 99-076, 99-077                               |
+----------------+-------------+----------------------------------------------+
| DFRC 99        | 06/30/1999- | 99-091, 99-090, 99-087, 99-086, 99-085,      |
|                | 10/19/1999  | 00-013, 00-011, 00-008                       |
+----------------+-------------+----------------------------------------------+
| WINTEX         | 03/15/1999- | 99-050, 99-051, 99-053, 99-054, 99-055,      |
|                | 04/03/1999  | 99-056, 99-057, 99-058, 99-059, 99-060       |
+----------------+-------------+----------------------------------------------+
| TRMM-LBA       | 01/22/1999- | 99-029, 99-030, 99-031, 99-032, 99-033,      |
|                | 02/23/1999  | 99-034, 99-037, 99-038, 99-039, 99-040,      |
|                |             | 99-042, 99-043, 99-044, 99-045               |
+----------------+-------------+----------------------------------------------+
| DFRC 98        | 03/09/1998- | 98-031, 98-032, 98-033, 98-036, 98-040,      |
|                | 01/04/1999  | 98-041, 98-043, 98-078, 98-079, 98-080,      |
|                |             | 99-017, 99-018, 99-019, 99-020, 99-022,      |
|                |             | 99-023, 99-024                               |
+--------------+-------------+------------------------------------------------+
| Wallops 98     | 07/11/1998- | 98-086, 98-087, 98-088, 98-089, 98-090       |
|                | 07/16/1998  |                                              |
+----------------+-------------+----------------------------------------------+
| FIRE-ACE       | 05/13/1998- | 98-063, 98-064, 98-065, 98-066, 98-067,      |
|                | 06/08/1998  | 98-068, 98-069, 98-070, 98-071, 98-072,      |
|                |             | 98-073, 98-074, 98-075, 98-07698_077         |
+----------------+-------------+----------------------------------------------+
| Wallops 97     | 08/01/1997- | 97-135, 97-136, 97-137, 97-138, 97-139,      |
|                | 08/21/1997  | 97-140, 97-141, 97-142                       |
+----------------+-------------+----------------------------------------------+
| ARC 97         | 03/03/1997- | 97-062, 97-063, 97-064, 97-113, 97-114,      |
|                | 07/30/1997  | 97-126, 97-127, 97-128, 97-133               |
+----------------+-------------+----------------------------------------------+
| WINCE          | 01/28/1997- | 97-041, 97-042, 97-043, 97-044, 97-045,      |
|                | 02/13/1997  | 97-046, 97-047, 97-048, 97-049, 97-050       |
+----------------+-------------+----------------------------------------------+
| ARC 96         | 09/09/1996- | 96-176, 96-177, 97-012, 97-014               |
|                | 11/11/1996  |                                              |
+----------------+-------------+----------------------------------------------+
| Spokane 96     | 07/30/1996- | 96-156, 96-158, 96-159, 96-160, 96-162       |
|                | 08/20/1996  |                                              |
+----------------+-------------+----------------------------------------------+
| TARFOX         | 07/07/1996- | 96-145, 96-146, 96-147, 96-148, 96-149,      |
|                | 07/26/1996  | 96-150, 96-151, 96-152, 96-153, 96-154       |
+----------------+-------------+----------------------------------------------+
| Wallops 96     | 07/05/1996  | 96-144                                       |
+----------------+-------------+----------------------------------------------+
| ARC 96         | 04/02/1996- | 96-088, 96-089, 96-126, 96-127, 96-128       |
|                | 06/05/1996  |                                              |
+----------------+-------------+----------------------------------------------+
| SUCCESS        | 04/08/1996- | 96-100, 6-101, 96-102, 96-103, 96-104,       |
|                | 05/15/1996  | 96-105, 96-106, 96-107, 96-108, 96-109,      |
|                |             | 96-110, 96-111, 96-112, 96-113, 96-114,      |
|                |             | 96-115, 96-116, 96-117                       |
+----------------+-------------+----------------------------------------------+
| BOREAS 96      | 08/14/1996  | 96-161                                       |
+----------------+-------------+----------------------------------------------+
| ARESE          | 09/25/1995- | 95-175, 95-176, 95-197, 95-198, 95-199,      |
|                | 10/23/1995  | 96-001, 96-002, 96-003, 96-004, 96-006,      |
|                |             | 96-007, 96-008, 96-009, 96-010, 96-020       |
+----------------+-------------+----------------------------------------------+
| PR 95          | 09/21/1995- | 95-172, 95-173, 95-174                       |
|                | 09/23/1995  |                                              |
+----------------+-------------+----------------------------------------------+
| SCAR-B         | 08/13/1995- | 95-158, 95-160, 95-161, 95-162, 95-163,      |
|                | 09/11/1995  | 95-164, 95-165, 95-1665-167, 95-168, 95-169, |
|                |             | 95-170                                       |
+----------------+-------------+----------------------------------------------+
| ARC 95         | 06/19/1995- | 95-122, 95-123, 95-124, 95-125, 95-126,      |
|                | 08/10/1995  | 95-153, 95-156, 95-157                       |
+----------------+-------------+----------------------------------------------+
| ARMCAS         | 06/02/1995- | 95-112, 95-113, 95-115, 95-116, 95-117,      |
|                | 06/16/1995  | 95-118, 95-119, 95-120, 95-121               |
+----------------+-------------+----------------------------------------------+
| ALASKA-April95 | 03/29/1995- | 95-069, 95-070, 95-071, 95-072, 95-073,      |
|                | 04/25/1995  | 95-074, 95-075, 95-076, 95-077, 95-078,      |
|                |             | 95-079                                       |
+----------------+-------------+----------------------------------------------+

LAADS also has few other campaigns from 1993 and 1994 in its archive for users 
to download. Complete information regarding various MAS campaigns and missions 
can be found at:
http://mas.arc.nasa.gov/index.html 

*****
*AMS*
***** 

The Autonomous Modular Sensor (AMS) is an airborne scanning spectrometer that 
acquires high spatial resolution imagery of the Earth's features from its 
vantage point on-board low and medium altitude research aircraft. Data 
acquired by AMS are helping to define, develop, and test algorithms for use in 
a variety of scientific programs that emphasize the use of remotely sensed 
data to monitor variation in environmental conditions, assess global change, 
and respond to natural disasters.
 
The AMS data is now available through our anonymous FTP site on 
ladsweb.nascom.nasa.gov. The directory contains AMS data organized by flight 
number and file type. The file types include Level 1B (AMSL1B), Level 2 ice 
phase (AMSL2I), and Level 2 water phase (AMSL2W) and the flight numbers are 
13_920, 13_921, 13_923, 13_13_924, and 13_925. So, the directory structure is 
as follows:
 
/ AMS/<flight number>/<product>
 
For example, AMSL1B product files for flight number 13-920 can be found on 
ladsweb in the following directory:
 
AMS/13_920/AMSL1B
 
Information about the flights for which data are stored on ladsweb is given in 
the table below:
 
+--------+------------+-------------------------------------+-----------+
| Flight | Date       | Location                            | Status    |
+--------+------------+-------------------------------------+-----------+
| 13-925 | 02/06/2013 | Southern California / Pacific Ocean | Level-2   |
+--------+------------+-------------------------------------+-----------+
| 13-924 | 02/03/2013 | California / Pacific Ocean          | Level-2   |
+--------+------------+-------------------------------------+-----------+
| 13-923 | 02/01/2013 | Southern California / Pacific Ocean | Level-2   |
+--------+------------+-------------------------------------+-----------+
| 13-921 | 01/28/2013 | Southern California / Pacific Ocean | Level-2   |
+--------+------------+-------------------------------------+-----------+
| 13-920 | 01/22/2013 | Central California                  | Level-2   |
+--------+------------+-------------------------------------+-----------+
 
Information about the flights, along with other resources, can be found on the 
AMS home page at NASA Ames Research Center:
 
http://asapdata.arc.nasa.gov/ams
 
*******
*MERIS*
*******

The Medium Resolution Imaging Spectrometer (MERIS) is one of the instruments 
on board the European Space Agency (ESA) satellite ENVISAT. MERIS, a 
push-broom imaging spectrometer, measures the solar radiation reflected by the 
Earth at a ground spatial resolution of 300 m, in 15 spectral bands in the 
visible and near infrared wavelengths. MERIS allows global coverage of the 
Earth in 3 days.

The primary mission of MERIS is to monitor the ocean color including 
chlorophyll concentrations for open oceans and coastal areas, yellow total 
suspended matter. In addition, MERIS provides land parameter measurements like 
vegetation indices and atmospheric parameters like water vapor, cloud top 
pressure, cloud types, aerosol optical thickness, and more.

MERIS data are provided at 3 different levels of processing - Level 0, Level 1,
Level 2 - and at 3 different spatial resolutions - Full, Reduced and Low. They 
may also be sorted and visualized very easily using a browse product (RGB 
color coded images of full-resolution data). Child products, including a 
limited number of measurement data sets, may also be available in reduced and 
full resolution.

For the same image, a Full-Resolution (FR) image has 4 x 4 more points (pixels)
than the same image in Reduced Resolution (RR), and an RR image has 4 x 4 more 
points (pixels) than the same image in Low Resolution (LR). Accordingly, a 
pixel in a FR image represents a ground area of 260 m x 290 m, in a RR image 
it depicts an area of 1,040 m x 1,160 m, and in a LR image it covers an area 
of 4,160 m x 4,640 m. It is to note that the instrument always takes 
measurements with full resolution; i.e., 260 m x 290 m ground resolution and 
an onboard averaging generates the RR images, while LR images are generated 
during the ground processing by further averaging the RR data. 

The MERIS product StartTime and EndTime recorded in LAADS are taken from
the time stamps of the first and last line in the product (FIRST_LINE_TIME 
and LAST_LINE_TIME).  The SENSING_START and SENSING_STOP metadata in the 
main header are not reliable. More info on MERIS data is available at:

https://earth.esa.int/web/guest/missions/esa-operational-eo-missions/envisat/instruments/meris

The MERIS data is available publicly through LAADS Website at no cost to 
registered users who have agreed to the terms and conditions set by NASA and 
the European Space Agency (ESA) to access MERIS data. LAADS is providing 
following three MERIS products

1. Full Resolution, Full Swath Geolocated and Calibrated TOA Radiance 
   (Product ID = MER_FRS_1P).
2. Reduced Resolution Geolocated and Calibrated TOA Radiance (stripline) 
   (Product ID = MER_RR__1P).
3. Reduced Resolution Geophysical Product for Ocean, Land and Atmosphere 
   (stripline) (Product ID = MER_RR__2P).

The MERIS data are stored at LAADS Web only in a predefined directory 
structure based on Archiveset (AS),  Product ID,  Data-Year,  and Data-Day:

490/Product_ID/YYYY/DDD/.

In order to access MERIS data, user has to complete three following steps:

Step 1. Register for an EOSDIS user account through the EOSDIS User
        Registration System. If you already have an EOSDIS user account, 
        continue to the next step.

Step 2. Add authorization for the MERIS data to your EOSDIS user account by 
        entering your EOSDIS login information and filling out a short form.

Step 3. Visit https://ladsweb.nascom.nasa.gov/MERIS/ and enter your EOSDIS
        login information to access the MERIS data.
