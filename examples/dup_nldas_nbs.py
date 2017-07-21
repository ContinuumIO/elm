#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import copy
import os
import sys
import getpass

from six.moves.urllib.parse import urlparse
from six.moves import range
from lxml import etree
from pydap.cas.urs import setup_session

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



def dups_to_indexes(field_names):
    """Modify field_names list in-place, such that duplicates get assigned
    integers as if they're indexes. Return field_names.

    For example:
      >>> dups_to_indexes(['A', 'B', 'A'])
      ['A[0]', 'B', 'A[1]']
    """
    n_fields = len(field_names)
    for i in range(n_fields):
        ufn = field_names[i]
        ct = 1
        for j in range(i+1, n_fields):
            if field_names[j] == ufn:
                field_names[j] += '['+str(ct)+']'
                ct += 1
        if ct > 1:
            field_names[i] += '[0]'
    return field_names

def get_xml_data(metadata, root, prefix, sep='/'):
    """Fill metadata with attributes from XML tree referenced by root.
    Nested attributes are delimited by sep.
    """
    children = root.getchildren()
    if not children:
        value = (None if root.text is None or not root.text.strip()
                 else root.text.strip())
        metadata[prefix] = value
    field_names = dups_to_indexes([elt.tag for elt in children])
    for idx, child in enumerate(children):
        next_prefix = prefix+sep+field_names[idx]
        get_xml_data(metadata, child, next_prefix)
    return metadata

username = os.environ.get('NLDAS_USERNAME', raw_input('NLDAS Username: '))
passwd = os.environ.get('NLDAS_PASSWORD', getpass.getpass('Password: '))
session = setup_session(username, passwd)
os.environ['NLDAS_USERNAME'] = username
os.environ['NLDAS_PASSWORD'] = passwd
def get_metadata(url):
    """Return a dict describing data at URL (which points to XML file).
    """
    print('Downloading {}...'.format(url))
    sys.stdout.flush()
    resp = session.get(url)
    print('Parsing XML...')
    sys.stdout.flush()
    root = etree.fromstring(resp.content)
    return get_xml_data({}, root, root.tag)


first_nb = 'NLDAS_FOR0125_H.001.ipynb'
first_nb_json = json.load(open(first_nb))
for url in urls:
    new_nb_name = urlparse(url).path.split(os.sep)[3]
    dest_filename = new_nb_name + '.ipynb'
    if (os.path.isfile(dest_filename) and
            not (len(sys.argv) > 1 and sys.argv[1] in ('-f', '--force'))):
        print('Skipping {} (already exists)...'.format(dest_filename))
        continue

    metadata = get_metadata(url+'.xml')
    print('Creating new notebook...')
    new_nb_json = copy.deepcopy(first_nb_json)
    longname = metadata['S4PAGranuleMetaDataFile/CollectionMetaData/LongName']
    new_nb_json['cells'][0]['source'][0] = '# {}:\n  {}'.format(new_nb_name, longname)
    new_nb_json['cells'][3]['source'][0] = "url = '{}'\n".format(url)
    print('Writing {}...'.format(dest_filename))
    with open(dest_filename, 'w') as fp:
        json.dump(new_nb_json, fp)
