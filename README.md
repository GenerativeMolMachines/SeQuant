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

## Getting Started With API
On the [wiki page](https://github.com/GenerativeMolMachines/SeQuant/wiki/SequantAPI), we have listed documentation on how to use the API to get Sequant embeddings.

## Datasets description
### Benchmarking datasets
The .csv files named antidia, antiinf, antimic and antiox are datasets used for benchmarking process.
### Energy data
File named as energy_data.csv is a dataset that contains DFT-derived descriptors. It is essential for the proper functioning of the tool.
### Small datasets
small_train_df.csv and small_test_df.csv are files partly used in SeQuant learning, here they were used for obtaining physicochemical propeties of the peptides with external library.

