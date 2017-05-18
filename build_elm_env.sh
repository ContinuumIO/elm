export ELM_BUILD_DIR=`pwd -P`

# The following two lines could potentially
# be bypassed by installing earthio, but
# using build_earthio_env.sh allows the
# test data to be downloaded as well.
build_elm_env(){
    git clone http://github.com/ContinuumIO/earthio && cd earthio || return 1;
    if [ "$EARTHIO_VERSION" = "" ];then
        export EARTHIO_VERSION="master";
    fi
    git checkout $EARTHIO_VERSION || return 1;
    . build_earthio_env.sh || return 1;
    # End of earthio and test data related section
    if [ "$PYTHON_TEST_VERSION" = "" ];then
        echo FAIL - Must define PYTHON_TEST_VERSION environment variable such as 2.7, 3.5 or 3.6 - FAIL
    else
        conda update -n root conda || return 1;
        conda remove -n root conda-build;conda install -n root conda-build;
        conda remove elm &> /dev/null;
        pip uninstall -y elm &> /dev/null;
        cd $ELM_BUILD_DIR || return 1;
        conda build -c conda-forge conda.recipe --python $PYTHON_TEST_VERSION conda.recipe || return 1;
        conda install -c conda-forge --use-local elm || return 1;;
    fi
}

build_elm_env && source activate $EARTHIO_ENV_TEST && echo OK