# elm/docs

Setup the documentation environment:
```
cd docs
conda env create -n elm-docs
. activate elm-docs
```

Have the documentation get rebuilt automatically, and the browser automatically refreshed, on each file edit:
```
sphinx-view -c ./source/conf.py source
```
