# elm examples

## Setup

First time only:
```
git clone https://github.com/ContinuumIO/elm.git
cd elm/examples
python dl_examples_env.py
conda env create -f environment.yml
source activate elm-examples
python download_sample_data.py
```

After the above commands finish, there should be a `elm/examples/data` directory with relevant data files.

On repeat runs, simply type:
```
source activate elm-examples
```

To add new datasets, edit the `datasets.yml` file and re-execute `download_sample_data.py`.

## Running the notebook server

The first time only, setup the config and password:
```
jupyter notebook --generate-config
jupyter notebook password
```

Hitting <kbd>Enter</kbd> for the password should allow passwordless logins.
