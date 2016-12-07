Release Procedure
==========

- Ensure all tests pass.

- Tag commit and push to github

.. code-block:: bash

    git tag -a x.x.x -m 'Version x.x.x'
    git push upstream master --tags

- Build conda packages

Define platform/setup specific environment variables *(fill in with your specifics)*

.. code-block:: bash

    # Location of your conda install. For me it's `~/anaconda/`
    CONDA_DIR=~/anaconda/

    # Platform code. For me it's `osx-64`
    PLATFORM=osx-64

    # Version number of elm being released (e.g. 0.2.0)
    VERSION=0.2.0


.. code-block:: bash

    # requires conda-build (conda install conda-build)
    conda build conda.recipe/ --python 3.5 --no-anaconda-upload -c conda-forge

Next, `cd` into the folder where the builds end up.

.. code-block:: bash

    cd $CONDA_DIR/conda-bld/$PLATFORM

Use ``conda convert`` to convert over the missing platforms (skipping the one for
the platform you're currently on):

.. code-block:: bash

    conda convert --platform osx-64 elm-$VERSION*.tar.bz2 -o ../
    conda convert --platform linux-64 elm-$VERSION*.tar.bz2 -o ../
    conda convert --platform linux-32 elm-$VERSION*.tar.bz2 -o ../
    conda convert --platform win-64 elm-$VERSION*.tar.bz2 -o ../
    conda convert --platform win-32 elm-$VERSION*.tar.bz2 -o ../

Use ``anaconda upload`` to upload the build to the ``elm`` channel. This requires
you to be setup on `anaconda.org`, and have the proper credentials to push to
the ``elm`` channel.

.. code-block:: bash

    # requires anaconda-client (conda install anaconda-client)
    anaconda login
    anaconda upload $CONDA_DIR/conda-bld/*/elm-$VERSION*.tar.bz2 -u elm
- Repeat ``conda build`` and ``anaconda upload`` steps above for ``--python 3.4`` as well
- Write the release notes:

 1. Run ``git log`` to get a listing of all the changes
 2. Remove any covered in the previous release
 3. Summarize the rest to focus on user-visible changes and major new features
 4. Paste the notes into github, under *n* ``releases``, then ``Tags``, then ``Edit release notes``.
