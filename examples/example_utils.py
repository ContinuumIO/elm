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

if 0:
    netrc_fpath = os.path.join(os.environ['HOME'], '.netrc')
    if not os.path.isfile(netrc_fpath):
        username = os.environ.get('NLDAS_USERNAME', raw_input('NLDAS Username: '))
        passwd = os.environ.get('NLDAS_PASSWORD', getpass.getpass('Password: '))
        with open(netrc_fpath, 'w') as fp:
            fp.write('machine urs.earthdata.nasa.gov login {} password {}'.format(username, passwd))
        os.chmod(netrc_fpath, 0o600)

from pydap.cas.urs import setup_session
username = os.environ.get('NLDAS_USERNAME', raw_input('NLDAS Username: '))
passwd = os.environ.get('NLDAS_PASSWORD', getpass.getpass('Password: '))
session = setup_session(username, passwd)

def dl_file(url):
    if 0:
        import pycurl
        from io import BytesIO
        buffer = BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/')
        c.setopt(c.WRITEDATA, buffer)
        c.perform()
        c.close()
        return buffer.getvalue()
    return session.get(url).content

def get_metadata(url):
    """Return a dict describing data at URL (which points to XML file).
    """
    print('Downloading {}...'.format(url))
    sys.stdout.flush()
    contents = dl_file(url)
    print('Parsing XML...')
    sys.stdout.flush()
    root = etree.fromstring(contents)
    return get_xml_data({}, root, root.tag)
