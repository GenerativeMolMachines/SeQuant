import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, '/nfs/home/enam/SeQuant')

from app.sequant_tools import SequantTools

from peptides import Peptide

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from sklearn.preprocessing import MinMaxScaler

print("imports done")

# Data import
df_train = pd.read_csv('../../utils/data/small_train_df.csv')
df_test = pd.read_csv('../../utils/data/small_test_df.csv')

df_train = df_train.drop(columns=['cluster_label', 'combined_cluster'], axis=1)
df_test = df_test.drop(columns=['cluster_label', 'combined_cluster'], axis=1)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# Retrieving descriptors by using SeQuant
scaler = MinMaxScaler(feature_range=(-1, 1))

sequences_train = df_train['sequence'].copy()
sequences_test = df_test['sequence'].copy()

max_peptide_length = 96
polymer_type = 'protein'

# Retrieving train descriptors
sqt_train = SequantTools(
    sequences=sequences_train,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_train = sqt_train.generate_latent_representations()
descriptors_normalized_train = pd.DataFrame(scaler.fit_transform(descriptors_train), columns=descriptors_train.columns)

# Retrieving test descriptors
sqt_test = SequantTools(
    sequences=sequences_test,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_test = sqt_test.generate_latent_representations()
descriptors_normalized_test = pd.DataFrame(scaler.fit_transform(descriptors_test), columns=descriptors_test.columns)

sequences_train.reset_index(drop=True, inplace=True)
descriptors_normalized_train.reset_index(drop=True, inplace=True)
sequences_test.reset_index(drop=True, inplace=True)
descriptors_normalized_test.reset_index(drop=True, inplace=True)

train = pd.concat([sequences_train, descriptors_normalized_train], axis=1)
test = pd.concat([sequences_test, descriptors_normalized_test], axis=1)

print('Descriptors have been received')
print("\n")
print(test)


# Getting physicochemical properties
def compute_properties(sequence):
    peptide = Peptide(sequence)
    return {
        "Aliphatic Index": peptide.aliphatic_index(),
        "Instability Index": peptide.instability_index(),
        "Theoretical Net Charge": peptide.charge(),
        "Isoelectric Point": peptide.isoelectric_point(),
        "Molecular Weight": peptide.molecular_weight()
    }


properties_train = train['sequence'].apply(compute_properties).apply(pd.Series)
properties_test = test['sequence'].apply(compute_properties).apply(pd.Series)

result_train = pd.concat([train, properties_train], axis=1)
result_test = pd.concat([test, properties_test], axis=1)

result_train = result_train.drop(columns=['sequence'], axis=1)
result_train.reset_index(drop=True, inplace=True)

result_test = result_test.drop(columns=['sequence'], axis=1)
result_test.reset_index(drop=True, inplace=True)

# Create a directory to save models if it doesn't already exist
os.makedirs('models', exist_ok=True)

# List of models and their short names for use in file names
models = {
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'RandomForest': RandomForestRegressor()
}

# List of properties to predict
properties = [
    "Aliphatic Index",
    "Instability Index",
    "Theoretical Net Charge",
    "Isoelectric Point",
    "Molecular Weight"
]

# For storing metrics
metrics_list = []

for model_name, model in models.items():
    for prop in properties:
        # Train the model
        model.fit(result_train.drop(columns=properties), result_train[prop])

        # Create a copy of the test set without the prediction columns
        test_features = result_test.drop(columns=properties).copy()

        # Predict on the test set
        predictions = model.predict(test_features)

        # Save the model
        model_filename = f'models/{model_name}_{prop.replace(" ", "_")}.pkl'
        joblib.dump(model, model_filename)

        # Add predictions to the test dataset
        result_test[f'{prop.replace(" ", "_")}_predicted'] = predictions

        # Calculate metrics
        mae = mean_absolute_error(result_test[prop], predictions)
        mse = mean_squared_error(result_test[prop], predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(result_test[prop], predictions)

        # Save metrics to the list
        metrics_list.append({
            'Model': model_name,
            'Property': prop,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })
        print(f'Property {prop} for model {model_name} has been predicted')
    print(f'Evaluation of model {model_name} is finished')

# Save metrics to a CSV file
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('benchmarking_results/model_metrics.csv', index=False)

# Save the test dataset with predictions
result_test.to_csv('benchmarking_results/result_test_with_predictions.csv', index=False)
