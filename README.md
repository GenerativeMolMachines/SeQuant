# SeQuant
Package for generating embeddings using the author's neural network.
## Setup
```
pip install requirements.txt
```
* pandas == 2.2.2
* numpy == 1.26.4
* requests == 2.32.3
* json == 2.0.9
* sklearn == 1.5.2
* biopython == 1.84
* joblib == 1.4.2
* xgboost == 2.1.2

## Overview
* File [benchmarking_embeddings.ipynb](https://github.com/GenerativeMolMachines/SeQuant/blob/main/benchmarking_embeddings.ipynb) implements a pipeline for comparing embeddings for several benchmark datasets for classification task: prediction of the antimicrobial ([Cao et al. 2023](https://doi.org/10.1093/bib/bbad058)), anti-inflammatory ([Raza et al. 2023](https://doi.org/10.1021/acs.jcim.3c01563)), antidiabetic ([Chen, Huang, He 2022](https://doi.org/10.7717/peerj.13581)) and antioxidative ([Qin et al. 2023](https://doi.org/10.1016/j.compbiomed.2023.106591)) peptides. 
* Datasets for playback are listed in the [data](https://github.com/GenerativeMolMachines/SeQuant/tree/main/data) directory. 

## Getting Started With API
On the [wiki page](https://github.com/GenerativeMolMachines/SeQuant/wiki/SequantAPI), we have listed documentation on how to use the API to get Sequant embeddings.

## Datasets description
### Benchmarking datasets
The .csv files named antidia, antiinf, antimic and antiox are datasets used for benchmarking process.
### Energy data
File named as energy_data.csv is a dataset that contains DFT-derived descriptors. It is essential for the proper functioning of the tool.
### Small datasets
small_train_df.csv and small_test_df.csv are files partly used in SeQuant learning, here they were used for obtaining physicochemical propeties of the peptides with external library.

