import numpy as np
import pandas as pd
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# Data import
stability_train_df = pd.read_csv('../utils/data/stability_train.csv')
stability_test_df = pd.read_csv('../utils/data/stability_test.csv')

# Filtration based on max length = 96
stability_train_df_preprocessed = stability_train_df[stability_train_df['seq'].apply(lambda x: len(x)) <= 96]
stability_test_df_preprocessed = stability_test_df[stability_test_df['seq'].apply(lambda x: len(x)) <= 96]

# Filtration based on known monomers
known_monomers = set(''.join(monomer_smiles.keys()))
filtered_train_df = stability_train_df_preprocessed[~stability_train_df_preprocessed['seq'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]
filtered_test_df = stability_test_df_preprocessed[~stability_test_df_preprocessed['seq'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]

# Retrieving descriptors by using SeQuant
sequences_train = stability_train_df['seq']
sequences_test = stability_test_df['seq']
max_peptide_length = 96
polymer_type = 'protein'

sqt_train = SequantTools(
    sequences=sequences_train,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_train = sqt_train.generate_latent_representations()

sqt_test = SequantTools(
    sequences=sequences_test,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_test = sqt_test.generate_latent_representations()

# Predictions
targets_train = filtered_train_df['label']
targets_test = filtered_test_df['label']

X_train, X_test, y_train, y_test = descriptors_train, descriptors_test, targets_train, targets_test

"""Models' list"""
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'SVR': SVR(),
    'KNeighborsRegressor': KNeighborsRegressor()
}

"""Function for model evaluation"""


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    spearman_corr, _ = spearmanr(y_test, predictions)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Spearman': spearman_corr
    }


"""Evaluation"""
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

"""Results"""
for name, metrics in results.items():
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\n")

