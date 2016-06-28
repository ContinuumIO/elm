## Testing Types:

 * Small-data unit tests of dask / xarray related functions asserting same results as similar numpy code.  These may not be necessarily tied to an image - in some cases these will be just tests of classifiers on arrays of semi-random data.
 * Testing analyst workflows through several steps of the pipeline, inclusive of input/ output to file.  There are two types:
   * Artificial data
     * This may involve creating image files from semi-random data then doing an analyst workflow with results that are more predictable than real-world data. Examples:
     * A before-test patch creates images of 12 unique colors to validate a classifier with 12 classes, and then the same images with noise or a different number of classes
   * Real world data
     * Real world data Real world data tests may help us develop promotional material for the new image pipeline.

## Test Framework

Use `pytest`. Use `pytest.mark` to mark tests that need special resources, e.g. a test data set, or tests that are slow, e.g:

```python
import pytest
@pytest.mark.slow
def test_kmeans_reporting():
    # code using simple assert statements
```
Over time, we can figure out marks other than `slow` that pertain to data sets that have to be unpacked and/or downloaded in advance.

Where temporary directories are needed, try to use the `pytest` local fixture `tmpdir`, e.g.:

```python
def test_sgd_regressor_simple(tmpdir):
    test_dir = str(tmpdir.join('test_sgd_regressor_simple'))
    # code using test_dir as directory
```
## Continuous Integration

 * Travis and/or anaconda-build on push events of every branch
 * Eventually, a nightly test cycle on an ec2 machine for longer running tests, as needed (tests marked slow or too slow for every push event)
 *

## Test Directory Structure

Make a tests directory in each of the elm subpackages.
