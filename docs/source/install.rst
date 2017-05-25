Install ELM
===========

You can install elm with ``conda`` or by installing from source.

Install from Conda
~~~~~~~~~~~~~~~~~~

To install the latest release of ``elm`` and ``earthio`` do:

.. code-block:: bash

    conda create -c elm/label/dev -c elm -c conda-forge -c ioam -c conda-forge -c scitools/label/dev --name earth-env-35 python=3.5 elm earthio

Note the command above uses the ``elm`` organization on `anaconda.org<http://anaconda.org>`_ and the command uses the ``dev`` label within that organization to get ``elm`` and the default ``main`` label for ``earthio`` .

This installs ``elm`` and ``earthio`` and all common dependencies. The channel arguments shown above may change with build system refactoring over the next few months.  If you encounter any issues with installation of the latest release from ``conda`` or source, then raise a github issue `in the elm repo here <http://github.com/ContinuumIO/elm/issues>`_ or email psteinberg [at] continuum [dot] io.

Install from Source
~~~~~~~~~~~~~~~~~~~

To install ``elm`` from `source <https://github.com/ContinuumIO/elm>`_ and quick check the install:

.. code-block:: bash

    git clone https://github.com/ContinuumIO/elm
    cd elm
    export ELM_EXAMPLE_DATA_PATH=~/elm-data
    PYTHON_TEST_VERSION=3.5 EARTHIO_INSTALL_METHOD=conda . build_elm_env.sh
    python -c "from earthio import *;from elm import *"

Add the following ``ELM_EXAMPLE_DATA_PATH`` to your .bashrc or environment to avoid >1 download of the test data and have the test data discovered by ``py.test``:

.. code-block:: bash

    export ELM_EXAMPLE_DATA_PATH=~/elm-data

Do the tutorials and examples:

 * :doc:`K-Means with LANDSAT example<clustering_example>`
 * :doc:`Examples <examples>`
