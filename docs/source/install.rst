Install ELM
===========

You can install elm with ``conda`` or by installing from source.

Install from Conda
~~~~~~

To install the latest release of elm

.. code-block:: bash

    conda install -c elm -c conda-forge elm elm 

This installs elm and all common dependencies. The channel arguments (``c elm -c conda-forge`` ) are typically required.


Install from Source
~~~~~~

To install elm from source, clone the repository from `github
<https://github.com/ContinuumIO/elm>`_:

.. code-block:: bash

    git clone https://github.com/ContinuumIO/elm.git
    cd elm
    conda env create
    source activate elm-env
    python setup.py develop

Clone the ``elm-data`` repo using Git LFS so that more tests can be run:

.. code-block:: bash

    brew install git-lfs # or apt-get, yum, etc
    git lfs install
    git clone https://github.com/ContinuumIO/elm-data
    git remote add origin https://github.com/ContinuumIO/elm-data

Add the following to your .bashrc or environment, changing the path depending on where you have cloned ``elm-data``:

.. code-block:: bash

    export ELM_EXAMPLE_DATA_PATH=/Users/peter/Documents/elm-data

Do the tutorials and examples:

 * :doc:`K-Means with LANDSAT example<clustering_example>`
 * :doc:`Examples <examples>`
