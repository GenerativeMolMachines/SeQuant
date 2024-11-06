import pandas as pd
import numpy as np

import sys

sys.path.insert(0, '/nfs/home/enam/SeQuant')
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
print("imports done")

# Data import
df = pd.read_csv('../../utils/data/DB_hyperfiltered.csv')
df['Class'] = 1

# Filtration based on max length = 96
df_preprocessed = df[df['Sequence'].apply(lambda x: len(x)) <= 96]

# Filtration based on known monomers
known_monomers = set(''.join(monomer_smiles.keys()))
filtered_df = df_preprocessed[~df_preprocessed['Sequence'].apply(
    lambda x: any(monomer not in known_monomers for monomer in x)
)]

functions = filtered_df['Function'].unique()
print(len(functions))
print(functions)

print(len(filtered_df[filtered_df['Function'] == 'Opioid']))
# Retrieving descriptors by using SeQuant
scaler = MinMaxScaler(feature_range=(-1, 1))

sequences = filtered_df['Sequence']
max_peptide_length = 96
polymer_type = 'protein'

sqt = SequantTools(
    sequences=sequences,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors = sqt.generate_latent_representations()
descriptors_normalized = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns)

final_df = pd.concat([filtered_df, descriptors_normalized], axis=1)
print(final_df)
print('Descriptors have been normalized')
print("\n")

# Define the parameter grid for each model
param_grids = {
    'LogReg': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']},
    'SVC': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
    'kNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
}

# Models' list
models = {
    'LogReg': LogisticRegression(class_weight='balanced'),
    'SVC': SVC(class_weight='balanced', probability=True),  # Set probability=True for ROC AUC score
    'kNN': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(class_weight='balanced')
}


def evaluate_model(model, param_grid, functions, dataset):
    results_dict = {}

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)

    for i in range(len(functions)):
        function = functions[i]
        dataset['Class'] = 1
        dataset.loc[dataset["Function"] == function, "Class"] = 0
        labels = dataset["Class"]
        descriptors = dataset.drop(columns=['Sequence', 'Function', 'Class'])

        X_train, X_test, y_train, y_test = train_test_split(descriptors, labels, stratify=labels, test_size=0.3,
                                                            random_state=42)

        if len(set(y_test)) < 2:
            print(f"Skipping function {function} due to insufficient class diversity in test set.")
            continue

        # Fit the model using grid search
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_model = grid_search.best_estimator_

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities) if probabilities is not None else None
        mcc = matthews_corrcoef(y_test, predictions)

        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'MCC': mcc,
            'Best Params': grid_search.best_params_
        }

        results_dict[function] = results
        print(f"Model evaluated for function {function} with best parameters: {grid_search.best_params_}")

    return results_dict


# Evaluation on test set
final_results = {}
results_list = []

for name, model in models.items():
    print(f"Evaluating model: {name}")
    final_results[name] = evaluate_model(model, param_grids[name], functions, final_df)

    for function, metrics in final_results[name].items():
        metrics.update({'Model': name, 'Function': function})
        results_list.append(metrics)

df_results = pd.DataFrame(results_list)
df_results.to_csv('benchmarking_results/010.csv', index=False)
final_df.to_csv('benchmarking_results/embeddings_010.csv', index=False)

# Print the results on test set
print('EVALUATION ON TEST SET')
for name, predictions in final_results.items():
    print(f"Predictions for {name}:")
    for function, results in predictions.items():
        print(f"Results for {function}")
        for metric_name, value in results.items():
            if value is not None:
                print(f"{metric_name}: {value}")
        print("n")
    print("n")
