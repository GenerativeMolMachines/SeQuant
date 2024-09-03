import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/enam/SeQuant')
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

print("imports done")

# Data import
stability_train_df = pd.read_csv('../utils/data/stability_train.csv')
stability_test_df = pd.read_csv('../utils/data/stability_test.csv')
stability_valid_df = pd.read_csv('../utils/data/stability_valid.csv')

# Filtration based on max length = 96
stability_train_df_preprocessed = stability_train_df[stability_train_df['seq'].apply(lambda x: len(x)) <= 96]
stability_test_df_preprocessed = stability_test_df[stability_test_df['seq'].apply(lambda x: len(x)) <= 96]
stability_valid_df_preprocessed = stability_valid_df[stability_valid_df['seq'].apply(lambda x: len(x)) <= 96]

# Filtration based on known monomers
known_monomers = set(''.join(monomer_smiles.keys()))
filtered_train_df = stability_train_df_preprocessed[~stability_train_df_preprocessed['seq'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]
filtered_test_df = stability_test_df_preprocessed[~stability_test_df_preprocessed['seq'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]
filtered_valid_df = stability_valid_df_preprocessed[~stability_valid_df_preprocessed['seq'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]

# Retrieving descriptors by using SeQuant
sequences_train = stability_train_df['seq']
sequences_test = stability_test_df['seq']
sequences_valid = stability_valid_df['seq']
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

sqt_valid = SequantTools(
    sequences=sequences_valid,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_valid = sqt_valid.generate_latent_representations()

# Predictions
targets_train = filtered_train_df['label']
targets_test = filtered_test_df['label']
targets_valid = filtered_valid_df['label']

X_train, X_test, X_valid, y_train, y_test, y_valid = (
    descriptors_train, descriptors_test, descriptors_valid, targets_train, targets_test, targets_valid)

# Models' list
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'SVR': SVR(),
    'KNeighborsRegressor': KNeighborsRegressor()
}

# Optimization params
param_grids = {
    'LinearRegression': {},
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'KNeighborsRegressor': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
}


def optimize_model_with_grid_search(model, param_grid, X_train, X_valid, y_train, y_valid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    mse = mean_squared_error(y_valid, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid, predictions)
    spearman_corr, _ = spearmanr(y_valid, predictions)

    return {
        'Best Params': grid_search.best_params_,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Spearman': spearman_corr
    }


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


# Models optimization
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    results[name] = optimize_model_with_grid_search(model, param_grids[name], X_train, X_valid, y_train, y_valid)

# Results of optimization
for name, metrics in results.items():
    print(f"Results of optimization for {name}:")
    for metric_name, value in metrics.items():
        if metric_name == 'Best Params':
            print(f"{metric_name}: {value}")
        else:
            print(f"{metric_name}: {value:.4f}")
    print("\n")


results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train+X_valid, X_test, y_train+y_valid, y_test)

# Final results
for name, metrics in results.items():
    print('FINAL RESULTS')
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\n")

