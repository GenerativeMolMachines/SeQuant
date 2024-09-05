import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/enam/SeQuant') 
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
print("imports done")

random_state = 2024

# Data import
hemo_neg_df = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/hemo_neg.csv')
hemo_neg_df['label'] = 0

hemo_pos_df = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/hemo_pos.csv')
hemo_pos_df['label'] = 1

# Data preprocessing
hemo_df = pd.concat([hemo_neg_df, hemo_pos_df], ignore_index=True)
hemo_df = hemo_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

hemo_df = hemo_df.iloc[:, :500]

print("Data has been imported")
print("\n")


def data_filtration(data):
    # Filtration based on max length = 96
    data = data[data['seq'].apply(lambda x: len(x)) <= 96]

    # Filtration based on known monomers
    known_monomers = set(''.join(monomer_smiles.keys()))
    filtered_data = data[~data['seq'].apply(
        lambda x: any(monomer not in known_monomers for monomer in x)
    )]

    return filtered_data


hemo_df_filtred = data_filtration(hemo_df)

# Retrieving descriptors by using SeQuant
sequences_hemo = hemo_df_filtred['seq']
max_peptide_length = 96
polymer_type = 'protein'

sqt_hemo = SequantTools(
    sequences=sequences_hemo,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_hemo = sqt_hemo.generate_latent_representations()
descriptors_hemo.to_csv('/nfs/home/enam/SeQuant/app/utils/data/descriptors_hemo.csv', index=False)
del descriptors_hemo

print("Descriptors have been received")
print("\n")

# Predictions
targets_hemo = hemo_df_filtred['label']
descriptor_hemo = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/descriptors_hemo.csv')
print('Descriptors have been read')
print("\n")

# Stratified split
X_train_val_hemo, X_test_hemo, y_train_val_hemo, y_test_hemo = train_test_split(
    descriptor_hemo, targets_hemo, test_size=0.1, stratify=targets_hemo, random_state=random_state)
X_train_hemo, X_val_hemo, y_train_hemo, y_val_hemo = train_test_split(
    X_train_val_hemo, y_train_val_hemo, test_size=0.1, stratify=y_train_val_hemo, random_state=random_state)

print("Sets have been split")
print("\n")

# Models' list
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC(probability=True),
    'KNeighborsClassifier': KNeighborsClassifier()
}

# Optimization params
param_grids = {
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7]
    }
}


def optimize_model_with_grid_search(model_name, model, X_train, X_val, y_train, y_val):
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_val)
    probabilities = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, "predict_proba") else None

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    roc_auc = roc_auc_score(y_val, probabilities) if probabilities is not None else None

    return {
        'Best Params': grid_search.best_params_,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }, best_model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities) if probabilities is not None else None

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }


print('Optimization of HEMO dataset started')
print("\n")

# Optimization with grid search on hemo dataset
results_hemo = {}
best_models_hemo = {}
counter = 1
for name, model in models.items():
    results_hemo[name], best_models_hemo[name] = optimize_model_with_grid_search(name, model, X_train_hemo, X_val_hemo,
                                                                                 y_train_hemo, y_val_hemo)
    print(f'model {counter} passed')
    counter += 1

# Results for hemo dataset
print('HEMOLYSIS DATASET')
for name, metrics in results_hemo.items():
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name}: {value}")
    print("\n")

print('Evaluation of HEMO ds')
print("\n")
# Final evaluation on test set using the best models
final_results_hemo = {}
for name, model in best_models_hemo.items():
    final_results_hemo[name] = evaluate_model(model, X_train_hemo + X_val_hemo, X_test_hemo, y_train_hemo + y_val_hemo,
                                              y_test_hemo)

# Print final results on test set
print('FINAL EVALUATION ON TEST SET (HEMOLYSIS DATASET)')
for name, metrics in final_results_hemo.items():
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name}: {value:.4f}")
    print("\n")
