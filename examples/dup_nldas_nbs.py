#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import copy
import os
from six.moves.urllib.parse import urlparse

urls = [
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FOR0125_M.001/2007/NLDAS_FOR0125_M.A200712.001.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FOR0125_MC.001/NLDAS_FOR0125_MC.ACLIM12.001.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.002/2017/197/NLDAS_FORA0125_H.A20170716.1200.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201706.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_MC.002/NLDAS_FORA0125_MC.ACLIM12.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORB0125_H.002/2017/197/NLDAS_FORB0125_H.A20170716.1200.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORB0125_M.002/2017/NLDAS_FORB0125_M.A201706.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORB0125_MC.002/NLDAS_FORB0125_MC.ACLIM12.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_MOS0125_H.002/2017/197/NLDAS_MOS0125_H.A20170716.1200.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_MOS0125_M.002/2017/NLDAS_MOS0125_M.A201706.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_MOS0125_MC.002/NLDAS_MOS0125_MC.ACLIM12.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_H.002/2017/197/NLDAS_NOAH0125_H.A20170716.0000.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_M.002/2017/NLDAS_NOAH0125_M.A201706.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_MC.002/NLDAS_NOAH0125_MC.ACLIM12.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_VIC0125_H.002/2017/196/NLDAS_VIC0125_H.A20170715.2300.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_VIC0125_M.002/2017/NLDAS_VIC0125_M.A201706.002.grb',
    'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_VIC0125_MC.002/NLDAS_VIC0125_MC.ACLIM12.002.grb',
]


first_nb = 'NLDAS_FOR0125_H.001.ipynb'
first_nb_json = json.load(open(first_nb))
for url in urls:
    new_nb_name = urlparse(url).path.split(os.sep)[3]
    dest_filename = new_nb_name + '.ipynb'
    if os.path.isfile(dest_filename):
        print('Skipping {} (already exists)...'.format(dest_filename))
    else:
        new_nb_json = copy.deepcopy(first_nb_json)
        new_nb_json['cells'][0]['source'][0] = '# {}\n'.format(new_nb_name)
        new_nb_json['cells'][3]['source'][0] = "url = '{}'\n".format(url)
        print('Writing {}...'.format(dest_filename))
        with open(dest_filename, 'w') as fp:
            json.dump(new_nb_json, fp)
