#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import copy
import os
import sys
import getpass
from collections import OrderedDict
from functools import partial

import requests
from six.moves.urllib.parse import urlparse
from six.moves import range, input
from lxml import etree, html
from ipywidgets import widgets, Layout
from IPython.display import display, Javascript


PYCURL = True

if not PYCURL:
    from pydap.cas.urs import setup_session
    session = setup_session(
        os.environ.get('NLDAS_USERNAME') or input('NLDAS Username: '),
        os.environ.get('NLDAS_PASSWORD') or getpass.getpass('Password: ')
    )

def get_request(url, outfpath=None):
    global PYCURL
    if PYCURL:
        # outfpath must be set
        import pycurl
        from io import BytesIO
        buffer = BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, buffer)
        c.setopt(c.COOKIEJAR, '/tmp/cookie.jar')
        c.setopt(c.NETRC, True)
        c.setopt(c.FOLLOWLOCATION, True)
        #c.setopt(c.REMOTE_NAME, outfpath)
        c.perform()
        c.close()
        return buffer.getvalue()
    resp = requests.get(url)
    return resp.text

def dl_file(url):
    data_fpath = urlparse(url).path.lstrip(os.sep)
    data_dpath = os.path.dirname(data_fpath)
    if not os.path.exists(data_fpath):
        if not os.path.isdir(data_dpath):
            os.makedirs(data_dpath)
        if PYCURL:
            with open(data_fpath, 'w') as outfp:
                outfp.write(get_request(url))
        else:
            resp = session.get(url)
            with open(data_fpath, 'w') as outfp:
                outfp.write(resp.content)
    return data_fpath

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

def get_metadata(url):
    """Return a dict describing data at URL (which points to XML file).
    """
    print('Downloading {}...'.format(url))
    sys.stdout.flush()
    #resp = requests.get(url)
    contents = dl_file(url)
    print('Parsing XML...')
    sys.stdout.flush()
    root = etree.fromstring(contents) #resp.text)
    return get_xml_data({}, root, root.tag)

class GRBSelector(object):
    def __init__(self, base_url='https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/', **layout_kwargs):
        self.base_url = base_url
        self.selected_url = None
        if 'min_width' not in layout_kwargs:
            layout_kwargs['min_width'] = '30%'
        self.label_layout = Layout(**layout_kwargs)

        dd = widgets.Select(
            options=self.get_links(
                base_url,
                href_filter=self.dir_and_not_data,
            ),
            description='', #urlparse(base_url).path,
        )

        dd.observe(partial(self.on_value_change, url=self.base_url), names='value')
        lbl = widgets.Label(urlparse(self.base_url).path, layout=self.label_layout)
        hbox = widgets.HBox([lbl, dd])
        self.elts = [hbox, lbl, dd]
        display(hbox)

    def on_value_change(self, change, url):
        next_url = change['new']
        if next_url is None: # 'Select...' chosen
            return
        if next_url.endswith('.grb'): # File reached
            return self.select_url(next_url)
        [w.close() for w in self.elts]
        links = self.get_links(next_url,
                               href_filter=(self.dir_and_not_data
                                            if next_url == self.base_url
                                            else self.dir_or_grib))
        if not links:
            return
        next_dd = widgets.Select(
            options=links,
            description='', #urlparse(url).path,
        )
        next_dd.observe(partial(self.on_value_change, url=next_url), names='value')
        lbl = widgets.Label(urlparse(next_url).path, layout=self.label_layout)
        hbox = widgets.HBox([lbl, next_dd])
        self.elts = [hbox, lbl, next_dd]
        display(hbox)

    def get_links(self, url, href_filter=None):
        progress = widgets.IntProgress(value=0, min=0, max=10, description='Loading:')
        display(progress)

        links = OrderedDict()
        links['Select an endpoint...'] = None
        if url != self.base_url:
            up_url = os.path.dirname(url.rstrip(os.sep))
            up_path = os.path.dirname(urlparse(url).path.rstrip(os.sep))
            if not up_url.endswith(os.sep):
                up_url += os.sep
            links['Up to {}...'.format(up_path)] = up_url
        if 0:
            resp = requests.get(url); progress.value += 1
            root = html.fromstring(resp.text); progress.value += 1
        else:
            contents = get_request(url); progress.value += 1
            root = html.fromstring(contents); progress.value += 1
        hrefs = root.xpath('body/table//tr/td/a/@href'); progress.value += 1
        parent_path = os.path.dirname(urlparse(url).path.rstrip(os.sep))
        for hrefct, href in enumerate(sorted(hrefs)):
            if hrefct % int(11 - progress.value) == 0:
                progress.value += 1
            if ((href_filter is not None and
                    not href_filter(href)) or
                urlparse(href).path.rstrip(os.sep).endswith(parent_path)):
                #print('filtered {} with {}'.format(href, href_filter))
                continue
            link_name = urlparse(href).path
            links[link_name] = url + href
        if len(links) <= 2:
            links = OrderedDict()

        progress.close()

        return links

    def dir_and_not_data(self, href):
        return href.endswith(os.sep) and not href.endswith('data/')

    def dir_or_grib(self, href):
        return href.endswith(os.sep) or href.endswith('.grb')

    def select_url(self, url):
        self.selected_url = url
        display(Javascript("""
            var run = false, current = $(this)[0];
            $.each(IPython.notebook.get_cells(), function (idx, cell) {
                if (!run && (cell.output_area === current)) {
                    run = true;
                } else if (cell.cell_type == 'code') {
                    cell.execute();
                }
            });
        """))
