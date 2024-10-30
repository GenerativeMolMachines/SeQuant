import pandas as pd
import numpy as np

import sys

sys.path.insert(0, '/nfs/home/enam/SeQuant')
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
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

# Models' list
models = {
    'XGB': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=False)
}

"""
def custom_train_test_split(descriptors, labels, test_size=0.2, random_state=None):
    # split data indices on positive and negative classes
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]

    # split each class on train and test sets
    pos_train_indices, pos_test_indices = train_test_split(
        positive_indices, test_size=test_size, random_state=random_state)
    neg_train_indices, neg_test_indices = train_test_split(
        negative_indices, test_size=test_size, random_state=random_state)

    # concatenate pos and neg indices
    train_indices = np.concatenate([pos_train_indices, neg_train_indices])
    test_indices = np.concatenate([pos_test_indices, neg_test_indices])

    # shuffle indices
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # form final train and test sets
    X_train = descriptors.iloc[train_indices]
    X_test = descriptors.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    return X_train, X_test, y_train, y_test
"""


def custom_train_test_split(descriptors, labels, test_size=0.2, random_state=None):
    # Set random seed
    np.random.seed(random_state)

    # split each class on train and test sets
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]

    # Determine test examples of positive and negative in test
    num_pos_test = int(len(positive_indices) * test_size)
    num_neg_test = int(len(negative_indices) * test_size)

    # Choose indices for test
    pos_test_indices = np.random.choice(positive_indices, size=num_pos_test, replace=False)
    neg_test_indices = np.random.choice(negative_indices, size=num_neg_test, replace=False)

    # Determine train indices as remain
    pos_train_indices = np.setdiff1d(positive_indices, pos_test_indices)
    neg_train_indices = np.setdiff1d(negative_indices, neg_test_indices)

    train_indices = np.concatenate([pos_train_indices, neg_train_indices])
    test_indices = np.concatenate([pos_test_indices, neg_test_indices])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Form final sets
    X_train = descriptors.iloc[train_indices]
    X_test = descriptors.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    return X_train, X_test, y_train, y_test


def evaluate_model(model, functions, dataset):
    results_dict = {}

    for i in range(len(functions)):
        function = functions[i]
        dataset['Class'] = 1
        dataset.loc[dataset["Function"] == function, "Class"] = 0  # set the class of interest to have label 0, the rest are 1
        labels = dataset["Class"]
        descriptors = dataset.drop(columns=['Sequence', 'Function', 'Class'])

        # Training
        X_train, X_test, y_train, y_test = custom_train_test_split(descriptors, labels, test_size=0.3, random_state=42)

        if len(set(y_test)) < 2:
            print(f"Skipping function {function} due to insufficient class diversity in test set.")
            continue

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

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
            'MCC': mcc
        }
        results_dict[function] = results
        print(f"Model evaluated for function {function}")

    return results_dict


# Evaluation on test set
final_results = {}
for name, model in models.items():
    print(f"Evaluating model: {name}")
    final_results[name] = evaluate_model(model, functions, final_df)

# Print the results on test set
print('EVALUATION ON TEST SET')
for name, predictions in final_results.items():
    print(f"Predictions for {name}:")
    for function, results in predictions.items():
        print(f"Results for {function}")
        for metric_name, value in results.items():
            if value is not None:
                print(f"{metric_name}: {value:.4f}")
        print("\n")
    print("\n")
