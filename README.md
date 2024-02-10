# UCLouvain - LACTU2110  - Modélisation prédictive et apprentissage statistique en assurance

## Pre-requisits

This code was build using R 4.3.2. 
If working on Windows, Rtools 4.3 is required to build some of the packages we will use.
We will use `renv` to handle the R package dependencies. 

We will also rely on pre-commit, which requires a python installation. This can be done using the following

- install miniconda if you don’t have it already: `reticulate::install_miniconda()`. This needs reticulate >= 1.14.
- install the pre-commit framework with `precommit::install_precommit()` into the conda environment r-precommit. Do not install other packages into this environment.

Then, in a fresh R session:

```
# once in every git repo either
# * after cloning a repo that already uses pre-commit or
# * if you want introduce pre-commit to this repo
precommit::use_precommit()
```

See also: https://lorenzwalthert.github.io/precommit/dev/articles/precommit.html

## Materiel du cours.

1. [A Brief introduction to R and Descriptive analysis of the dataset](1.%20Introduction/1.%20Brief%20Introduction%20to%20R%20and%20Descriptive%20Analysis%20of%20the%20Dataset.md)
2. [Tree-based models - CART](2.%20CART/2.%20Tree-based%20models%20-%20CART.md)

## Required Software

For Windows, Rtools : https://cran.r-project.org/bin/windows/Rtools/ (Make sure to check the box add to path when installing Rtools)

## Conda environment (optionnal)

If you want to use a conda environment, you can use the following commands to create a R environment, install jupyter notebook and allow use of R in jupyter notebook

### Short version 

You are located in the directory containing environment.yml from this repo:
`conda env create -f environment.yml`
An environment called `lactu2110` will be created.

Then, you can activate the environment with `conda activate lactu2110` and launch jupyter notebook with
`jupyter notebook`

### Long version

`conda create --name lactu2110 r-base`
`conda activate lactu2110`
`conda install jupyter`
`conda install -c r r-irkernel`

Then, you can launch jupyter notebook with
`jupyter notebook`
