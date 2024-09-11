import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/enam/SeQuant')
from app.utils.conctants import monomer_smiles

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

print("imports done")

# Data import
stability_train_label = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/stability_train.csv') # seqs and labels
stability_test_label = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/stability_test.csv') # seqs and labels

stability_train_descriptors = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/stability_train_pbm.csv') # seqs and labels
stability_test_descriptors = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/stability_test_pbm.csv') # seqs and labels

# Filtration based on max length = 96
stability_train_label_preprocessed = stability_train_label[stability_train_label['seq'].apply(lambda x: len(x)) <= 96]
stability_test_label_preprocessed = stability_test_label[stability_test_label['seq'].apply(lambda x: len(x)) <= 96]

# Filtration based on known monomers
known_monomers = set(''.join(monomer_smiles.keys()))
filtered_train_label = stability_train_label_preprocessed[~stability_train_label_preprocessed['seq'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]
filtered_test_label = stability_test_label_preprocessed[~stability_test_label_preprocessed['seq'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]

# Concatenation of descriptors and labels
stability_train_descriptors = stability_train_descriptors.drop_duplicates(subset='seq')
filtered_train_label = filtered_train_label.drop_duplicates(subset='seq')

stability_test_descriptors = stability_test_descriptors.drop_duplicates(subset='seq')
filtered_test_label = filtered_test_label.drop_duplicates(subset='seq')

stability_train_df = pd.merge(stability_train_descriptors, filtered_train_label, on='seq', how='left')
stability_test_df = pd.merge(stability_test_descriptors, filtered_test_label, on='seq', how='left')

stability_train_df = stability_train_df.dropna(how='any')
stability_test_df = stability_test_df.dropna(how='any')

stability_train_df.reset_index(drop=True, inplace=True)
stability_test_df.reset_index(drop=True, inplace=True)


# Predictions
scaler = MinMaxScaler(feature_range=(-1, 1))

targets_train = stability_train_df['label']
targets_test = stability_test_df['label']

descriptors_train = stability_train_df.drop(columns=['seq', 'label'])
descriptors_test = stability_test_df.drop(columns=['seq', 'label'])

descriptors_train = scaler.fit_transform(descriptors_train)
descriptors_test = scaler.fit_transform(descriptors_test)

X_train, X_test, y_train, y_test = (
    descriptors_train, descriptors_test, targets_train, targets_test
)

# Models' list
models = {
    'XGB': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(),
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

