Installation
============

There are two options for installing `elm`:

    - :ref:`conda-install`
    - :ref:`source-install`

.. _conda-install:

Install with Conda
~~~~~~~~~~~~~~~~~~

Conda is a package manager backed by `Continuum Analytics, Inc <http://continuum.io>`_. Notable features include first-class support for Python and R software that depends on C extensions, cross-platform support for Windows/Mac OSX/Linux, and standalone packages that are easy to distribute and deploy.

Stable
------

The "stable" release is more thoroughly tested, but does not include the latest experimental features:

.. code-block:: bash

    conda create -c elm -c conda-forge -c ioam -c scitools --name earth-env elm earthio

The above command creates a conda environment with the latest stable releases of `elm` and `earthio` installed. To begin using, activate the environment:

.. code-block:: bash

    source activate earth-env

If you encounter any issues with installation of the latest release from ``conda`` or source, then raise a github issue `in the elm repo here <http://github.com/ContinuumIO/elm/issues>`_ or email psteinberg [at] continuum [dot] io.

Development
-----------

The "development" releases are less stable, but include newer features:

.. code-block:: bash

    conda create -c elm/label/dev -c conda-forge -c ioam -c scitools/label/dev --name earth-env-dev python=3.5 elm earthio

Like for the stable release, activate the environment to begin using `elm`:

.. code-block:: bash

    source activate earth-env-dev

.. _source-install:

Install from Source
~~~~~~~~~~~~~~~~~~~

Installing `elm` from source is recommended if you want to develop `elm` features and iterate rapidly over your code changes. To install ``elm`` from `source <https://github.com/ContinuumIO/elm>`_:

.. code-block:: bash

    git clone https://github.com/ContinuumIO/elm
    cd elm
    export ELM_EXAMPLE_DATA_PATH=~/elm-data
    PYTHON_TEST_VERSION=3.5 EARTHIO_INSTALL_METHOD=conda . build_elm_env.sh

Verify the install with:

.. code-block:: bash

    python -c "from earthio import *;from elm import *"

You may want to add the following line to your .bashrc (or equivalent shell config) to avoid >1 download of the test data and have the test data discovered by ``py.test``:

.. code-block:: bash

    export ELM_EXAMPLE_DATA_PATH=~/elm-data

Do the tutorials and examples:

 * :doc:`K-Means with LANDSAT example<clustering_example>`
 * :doc:`Examples <examples>`
