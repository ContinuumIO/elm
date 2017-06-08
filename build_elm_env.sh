#!/bin/bash

set -e

export ELM_BUILD_DIR=`pwd -P`
export EARTHIO_VERSION="${EARTHIO_VERSION:-master}"

if [ \( "x$EARTHIO_INSTALL_METHOD" = "xconda" \) -o \( "x$EARTHIO_INSTALL_METHOD" = "xgit" \) ]; then
    rm -rf .earthio_tmp
    git clone http://github.com/ContinuumIO/earthio .earthio_tmp
    cd .earthio_tmp
    git fetch --all
    echo git checkout $EARTHIO_VERSION
    git checkout $EARTHIO_VERSION

    set +e
    IGNORE_ELM_DATA_DOWNLOAD=1 . build_earthio_env.sh
    set -e
else
    if [ ! -d $HOME/miniconda ]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
    fi
    export PATH="$HOME/miniconda/bin:$PATH"
    source deactivate
    conda env remove -n $EARTHIO_TEST_ENV || true
    conda env create -n $EARTHIO_TEST_ENV -f environment.yml
    conda install -n $EARTHIO_TEST_ENV $EARTHIO_CHANNEL_STR -c elm -y earthio
fi

source activate $EARTHIO_TEST_ENV

conda config --set always_yes true
conda config --set anaconda_upload no
conda install -n root conda conda-build

conda remove elm &> /dev/null || true
pip uninstall -y elm &> /dev/null || true
echo conda list is ------
conda list
echo conda "env" list is ------
conda env list

cd $ELM_BUILD_DIR
conda build $EARTHIO_CHANNEL_STR -c local --python $PYTHON --numpy $NUMPY conda.recipe
conda install $EARTHIO_CHANNEL_STR --use-local elm

set +e
