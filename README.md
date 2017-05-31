# NASA SBIR Phase I & II - Open Source Parallel Image Analysis and Machine Learning Pipeline

## Using the Code

To use this code:

#### Install:
Create the development environment:
```
conda env create
```

Activate the environment:
```
source activate elm-env
```
(older versions of the code may have `elm` in place of `elm-env` above.  The environment name was changed to avoid conflict with `elm` package on anaconda.org.  The `elm-env` is uploaded to the nasasbir org on anaconda.org.)

Install the source:
```
python setup.py develop
```
Clone the `elm-data` repo using Git LFS so that more tests can be run:
```
brew install git-lfs # or apt-get, yum, etc
git lfs install
git clone https://github.com/ContinuumIO/elm-data
git remote add origin https://github.com/ContinuumIO/elm-data
```

Add the following to your .bashrc or environment, changing the paths depending on where you have cloned elm-data:
```
export DASK_EXECUTOR=SERIAL
export ELM_EXAMPLE_DATA_PATH=/Users/psteinberg/Documents/elm-data
```

## Read more docs

```bash
cd docs
source activate elm-env
pip install recommonmark sphinx sphinx_rtd_theme numpydoc
make html
```
Then view the resulting files in docs/build in your `browser's` file:// protocol.
