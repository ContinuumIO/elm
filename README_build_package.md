## How to Build the `elm` Package

* Clone / checkout / pull the code you want to use from the repository
* `conda build conda.recipe --python 3.5 --no-anaconda-upload`
* Note where the built package is located from the output and copy it to upload command:

`anaconda upload -p elm /home/peter/miniconda/conda-bld/linux-64/elm-0.0.0-py35_0.tar.bz2 -u nasasbir --force`

This will upload the `elm` package under organization `nasasbir`, forcing replacement if the version exists.

## Conda Packaging

Note if new dependencies are added to environment.yaml, then they must also either appear in conda.recipe/meta.yaml as a conda requirement or conda.recipe/requirements.txt as a pip requirement.
