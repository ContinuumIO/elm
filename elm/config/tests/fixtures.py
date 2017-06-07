from __future__ import absolute_import, division, print_function, unicode_literals

import os

import pytest

if bool(int(os.environ.get('IS_TRAVIS', 1))):

    @pytest.fixture(autouse=True)
    def no_dask_distributed_on_travis(monkeypatch):
        monkeypatch.delattr("requests.sessions.Session.request")

else:
    @pytest.fixture(autouse=True)
    def no_dask_distributed_on_travis(monkeypatch):
        return
