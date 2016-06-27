import os

from iamlp.config.util import read_from_egg

METADATA_DIR = os.path.join('acquire','metadata')
EXAMPLE_LADSWEB_PRODUCTS = read_from_egg(os.path.join(METADATA_DIR,
                                         'example_ladsweb_products.txt'))

EXAMPLE_LADSWEB_PRODUCTS = {x[0].strip(): x[1].strip()
                            for x in EXAMPLE_LADSWEB_PRODUCTS.splitlines()}