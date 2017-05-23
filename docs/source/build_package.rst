How to Build the `elm` Package
==========

* Clone / checkout / pull the code you want to use from the repository

.. code-block:: bash

    conda build conda.recipe --python 3.5 --no-anaconda-upload

* Note where the built package is located from the output and copy it to upload command:

.. code-block:: bash

    conda build conda.recipe --python 3.5 --no-anaconda-upload
    anaconda upload -p elm /home/peter/miniconda/conda-bld/linux-64/elm-0.0.0-py35_0.tar.bz2 -u elm --force

This will upload the `elm` package under organization `nasasbir`, forcing replacement if the version exists.

Conda Packaging
-----------

Conda packaging systems are currently being updated for ``elm`` and ``earthio`` .  This help section will be updated soon (TODO).