# This file allows to convert Jupyter notebook files to Rmarkdown files.
# Rmarkdown files can then be run in Rstudio


list_of_files = c("1. Introduction/1. Brief Introduction to R and Descriptive Analysis of the Dataset.ipynb",
                  "2. CART/2. Tree-based models - CART.ipynb")

for (file in list_of_files){
  rmarkdown::convert_ipynb(input = file)
}


