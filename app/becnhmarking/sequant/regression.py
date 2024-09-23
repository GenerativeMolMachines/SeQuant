import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/enam/SeQuant')
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

print("imports done")

# Data import
stability_train_df = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/stability_train.csv')
stability_test_df = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/stability_test.csv')


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
scaler = MinMaxScaler(feature_range=(-1, 1))

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
descriptors_train = scaler.fit_transform(descriptors_train)

sqt_test = SequantTools(
    sequences=sequences_test,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_test = sqt_test.generate_latent_representations()
descriptors_test = scaler.fit_transform(descriptors_test)

# Predictions
targets_train = filtered_train_df['label']
targets_test = filtered_test_df['label']

X_train, X_test, y_train, y_test = (
    descriptors_train, descriptors_test, targets_train, targets_test
)

# Models' list
models = {
    'XGB': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(),
}


def plot_predictions(y_true, y_pred, title='Predictions vs True Values'):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.show()


def plot_distribution(y_true, y_pred, title='Distribution of Predictions and True Values'):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_true, color='blue', label='True Values', kde=True, stat="density", linewidth=0)
    sns.histplot(y_pred, color='red', label='Predictions', kde=True, stat="density", linewidth=0)
    plt.title(title)
    plt.legend()
    plt.show()


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions_test)
    mse = mean_squared_error(y_test, predictions_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions_test)
    spearman_corr, _ = spearmanr(y_test, predictions_test)

    plot_predictions(y_train, predictions_train, title='Train Predictions vs True Values')
    plot_distribution(y_train, predictions_train, title='Train Distribution of Predictions and True Values')

    plot_predictions(y_test, predictions_test, title='Test Predictions vs True Values')
    plot_distribution(y_test, predictions_test, title='Test Distribution of Predictions and True Values')

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Spearman': spearman_corr
    }


# Models evaluation
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

# Final results
for name, metrics in results.items():
    print('FINAL RESULTS')
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\n")

