How to Build the `elm` Package
==========

* Clone / checkout / pull the code you want to use from the repository

.. code-block:: bash 

    conda build conda.recipe --python 3.5 --no-anaconda-upload

* Note where the built package is located from the output and copy it to upload command:

.. code-block:: bash 

    conda build conda.recipe --python 3.5 --no-anaconda-upload
    anaconda upload -p elm /home/peter/miniconda/conda-bld/linux-64/elm-0.0.0-py35_0.tar.bz2 -u nasasbir --force

This will upload the `elm` package under organization `nasasbir`, forcing replacement if the version exists.

Upload elm-env Environment
-----------

Change directories to the repo clone dir:

.. code-block:: bash 
    anaconda upload -u nasasbir environment.yml

Later install with something like this,:

.. code-block:: bash 
    conda install nasasbir/elm-env

Conda Packaging
-----------

Note if new dependencies are added to environment.yaml, then they must also either appear in conda.recipe/meta.yaml as a conda requirement.  
