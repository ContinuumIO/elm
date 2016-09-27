## Using elm with Anaconda Cluster

* Follow the [install instructions for Anaconda cluster and Anaconda client](https://docs.continuum.io/anaconda-cluster/installation)
* Confim your install worked with:

`acluster`

* Copy the contents of this repo's `acluster/profiles` dir to: `~/.acluster/profiles` so that Anaconda cluster finds it
* Make sure you have an acluster provider `aws_east_nasasbir` or equivalent that is mentioned in the profile yaml files you copied. This involves modifying `~/.acluster/providers.yaml` to add `aws_east_nasasbir` as a provider, or modifying the elm cluster profile's `provider` to be your existing provider, such as a named bare metal cluster or different AWS region provider.
* Confirm you have the profile and providers consistent with each other, and your yaml format is okay:
  * `acluster list providers`
  * `acluster list profiles`
* Adjust the number of nodes and instance ID in the `elm` profile(s) as needed or create copies of the `elm` profile for different cluster configurations/sizes
* Create a cluster by referencing the profile name you have created, `elm` by default: 

`acluster create -p elm elm-cluster`
* Install geographic and image data analysis stack (after changing directories to the repo cloned locally):
```
acluster conda push environment.yml
```
* Install the `elm` machine learning code:
```
TOKEN=na-c671fafb-9323-43fd-9af7-ce7e2e640244
acluster conda create -c https://conda.anaconda.org/t/${TOKEN}/nasasbir -n elm python=3.5 elm --stream
```

(the `-c nasasbir` and `TOKEN` parts of the command above are related to privacy until the package is open source)

* Use the cluster's tools with:

```bash
acluster ssh
source activate elm-env
```
* See also the [list of packages installed as part of the `elm` environment](environment.yml)

### Dev Notes
If testing a cluster with new `elm` code, you may need to [build the elm package first](README_build_package.md)
