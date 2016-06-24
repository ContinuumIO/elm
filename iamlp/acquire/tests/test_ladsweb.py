
from iamlp.acquire.ladsweb_ftp import (login,
                                       download,
                                       get_sample_main,
                                       main,
                                       TOP_DIR,
                                       ftp_ls)

from iamlp.acquire import EXAMPLE_LADSWEB_PRODUCTS

def assert_can_login_and_ls():
    ftp = login()
    ftp.cwd(TOP_DIR)
    assert bool(ftp_ls())
    ftp.cwd('/'.join(TOP_DIR, tuple(EXAMPLE_LADSWEB_PRODUCTS.keys())[0]))
    assert bool(ftp_ls())


