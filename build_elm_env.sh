export ELM_BUILD_DIR=`pwd -P`

# The following two lines could potentially
# be bypassed by installing earthio, but
# using build_earthio_env.sh allows the
# test data to be downloaded as well.
build_elm_env(){
    rm -rf .earthio_tmp;
    git clone http://github.com/ContinuumIO/earthio .earthio_tmp && cd .earthio_tmp || return 1;
    if [ "$EARTHIO_VERSION" = "" ];then
        export EARTHIO_VERSION="master";
    fi
    git fetch --all || return 1;
    echo git checkout $EARTHIO_VERSION
    git checkout $EARTHIO_VERSION || return 1;
    conda config --set always_yes true;
    . build_earthio_env.sh && source activate $EARTHIO_TEST_ENV || return 1;
    conda config --set always_yes true;
    cd $ELM_BUILD_DIR || return 1;
    # End of earthio and test data related section
    if [ "$PYTHON" = "" ];then
        echo FAIL - Must define PYTHON environment variable such as 2.7, 3.5 or 3.6 - FAIL
        return 1;
    fi
    conda update -n root conda || return 1;
    conda remove -n root conda-build anaconda-client;
    conda config --set anaconda_upload no;
    conda remove elm &> /dev/null;
    pip uninstall -y elm &> /dev/null;
    cd $ELM_BUILD_DIR || return 1;
    echo conda list is ------
    conda list || return 1;
    echo conda "env" list is ------
    conda env list || return 1;
    conda build $EARTHIO_CHANNEL_STR --python $PYTHON conda.recipe || return 1;
    conda install $EARTHIO_CHANNEL_STR --use-local elm || return 1;
}

build_elm_env && source activate $EARTHIO_TEST_ENV && echo OK


