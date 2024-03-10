# UCLouvain - LACTU2110  - Modélisation prédictive et apprentissage statistique en assurance
## Materiel du cours.

1. [A Brief introduction to R and Descriptive analysis of the dataset](1.%20Introduction/1.%20Brief%20Introduction%20to%20R%20and%20Descriptive%20Analysis%20of%20the%20Dataset.md)
2. [Tree-based models - CART](2.%20CART/2.%20Tree-based%20models%20-%20CART.md)

## Required Software

For Windows, Rtools : https://cran.r-project.org/bin/windows/Rtools/ (Make sure to check the box add to path when installing Rtools)

## Conda environment (optional)

If you want to use a conda environment, you can use the following commands to create a R environment, install jupyter notebook and allow use of R in jupyter notebook

### Short version

You are located in the directory containing environment.yml from this repo:
`conda env create -f environment.yml`
An environment called `lactu2110` will be created.

Then, you can activate the environment with `conda activate lactu2110` and launch jupyter notebook with
`jupyter notebook`

### Long version

```python
conda create --name lactu2110 r-base
conda activate lactu2110
conda install jupyter
conda install -c r r-irkernel
```

Then, you can launch jupyter notebook with
`jupyter notebook`