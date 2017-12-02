#!/bin/bash

set -e

export ELM_BUILD_DIR=`pwd -P`

if [ ! -d "$HOME/miniconda" ]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    source deactivate
else
    source deactivate
    export PATH="$PATH:$(dirname $(which python))"
fi

conda config --set always_yes true
conda config --set anaconda_upload no
conda install -n root conda conda-build

# Create $TEST_ENV
conda env remove -n $TEST_ENV || true

cd $ELM_BUILD_DIR

conda remove -n root elm &> /dev/null || true
pip uninstall -y elm &> /dev/null || true

conda build $INSTALL_CHANNELS --python $PYTHON --numpy $NUMPY conda.recipe
conda create -n $TEST_ENV $INSTALL_CHANNELS --use-local python=$PYTHON numpy=$NUMPY elm
set +e
