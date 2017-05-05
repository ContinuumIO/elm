export ELM_BUILD_DIR=`pwd -P`

# The following two lines could potentially
# be bypassed by installing earthio, but
# using build_earthio_env.sh allows the
# test data to be downloaded as well.
git clone http://github.com/ContinuumIO/earthio && cd earthio
. build_earthio_env.sh
# End of earthio and test data related section
if [ "$EARTH_VERS" = "" ];then
    echo FAIL - Must define EARTH_VERS environment variable such as 2.7, 3.5 or 3.6 - FAIL
else
    conda update conda;
    conda install conda-build;
    conda remove elm &> /dev/null;
    pip uninstall -y elm &> /dev/null;
    conda build conda.recipe --python $EARTH_VERS
    conda install --use-local elm
fi

