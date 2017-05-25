export ELM_BUILD_DIR=`pwd -P`

# The following two lines could potentially
# be bypassed by installing earthio, but
# using build_earthio_env.sh allows the
# test data to be downloaded as well.
build_elm_env(){
    set -e
    rm -rf .earthio_tmp
    git clone http://github.com/ContinuumIO/earthio .earthio_tmp && cd .earthio_tmp
    # Temporary fix
    sed -i 's/PYTHON_TEST_VERSION/PYTHON/g' ./build_earthio_env.sh
    if [ "x$EARTHIO_VERSION" = "x" ]; then
        export EARTHIO_VERSION="master";
    fi
    git fetch --all
    echo git checkout $EARTHIO_VERSION
    git checkout $EARTHIO_VERSION
    set +e
    . build_earthio_env.sh && source activate $EARTHIO_TEST_ENV
    set -e
    cd $ELM_BUILD_DIR
    # End of earthio and test data related section
    if [ "x$PYTHON" = "x" ]; then
        echo FAIL - Must define PYTHON environment variable such as 2.7, 3.5 or 3.6 - FAIL
        return 1
    fi
    conda config --set always_yes true
    conda update -n root conda conda-build
    conda config --set anaconda_upload no
    conda remove elm &> /dev/null
    pip uninstall -y elm &> /dev/null
    cd $ELM_BUILD_DIR
    echo conda list is ------
    conda list
    echo conda "env" list is ------
    conda env list
    conda build $EARTHIO_CHANNEL_STR --python $PYTHON --numpy $NUMPY conda.recipe
    conda install $EARTHIO_CHANNEL_STR --use-local elm
    set +e
}

build_elm_env && source activate $EARTHIO_TEST_ENV && echo OK
