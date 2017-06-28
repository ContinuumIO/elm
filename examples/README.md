# elm examples

## Setup

```
git clone https://github.com/ContinuumIO/elm.git
cd elm/examples
python dl_examples_env.py
conda env create -f environment.yml
source activate elm-examples
```

The script (in the first line) downloads the necessary files from the examples directory in [datashader's repository](https://github.com/bokeh/datashader) so that elm can download the example data and rely on a superset of its _environment.yml_. The lines after that use the modified _environment.yml_ to create an env called _elm-examples_, which may be used for running the example notebooks.

## Running the notebook server

The first time only, setup the config and password:
```
jupyter notebook --generate-config
jupyter notebook password
```
Hitting <kbd>Enter</kbd> for the password should allow passwordless logins.

