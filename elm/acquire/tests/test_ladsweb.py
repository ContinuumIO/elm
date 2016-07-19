import pytest

from elm.acquire.ladsweb_ftp import (login,
                                       download,
                                       get_sample_main,
                                       main,
                                       TOP_DIR,
                                       ftp_ls)

from elm.acquire import EXAMPLE_LADSWEB_PRODUCTS

EXAMPLE = '/allData/3001/NPP_D17BRDFIP_L3/2012/137/'

@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.skip
def test_can_login_and_ls():
    ftp = login()
    ftp.cwd(EXAMPLE)
    ls = ftp_ls(ftp)
    assert any(x.endswith('hdf') for x in ls)


