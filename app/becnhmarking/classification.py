import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/enam/SeQuant') 
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
print("imports done")

random_state = 2024

# Data import
hemo_neg_df = pd.read_csv('../utils/data/hemo_neg.csv')
hemo_neg_df['label'] = 0

hemo_pos_df = pd.read_csv('../utils/data/hemo_pos.csv')
hemo_pos_df['label'] = 1

nf_neg_df = pd.read_csv('../utils/data/nf_neg.csv')
nf_neg_df['label'] = 0

nf_pos_df = pd.read_csv('../utils/data/nf_pos.csv')
nf_pos_df['label'] = 1

# Data preprocessing
hemo_df = pd.concat([hemo_neg_df, hemo_pos_df], ignore_index=True)
nf_df = pd.concat([nf_neg_df, nf_pos_df], ignore_index=True)

hemo_df = hemo_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
nf_df = nf_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


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
nf_df_filtred = data_filtration(nf_df)

# Retrieving descriptors by using SeQuant
sequences_hemo = hemo_df_filtred['seq']
sequences_nf = nf_df_filtred['seq']
max_peptide_length = 96
polymer_type = 'protein'

sqt_hemo = SequantTools(
    sequences=sequences_hemo,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_hemo = sqt_hemo.generate_latent_representations()

sqt_nf = SequantTools(
    sequences=sequences_nf,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein'
)

descriptors_nf = sqt_nf.generate_latent_representations()

# Predictions
targets_hemo = hemo_df_filtred['label']
targets_nf = nf_df_filtred['label']

"""Stratified split"""
X_train_hemo, X_test_hemo, y_train_hemo, y_test_hemo = train_test_split(
    descriptors_hemo, targets_hemo, test_size=0.2, stratify=targets_hemo, random_state=random_state)

X_train_nf, X_test_nf, y_train_nf, y_test_nf = train_test_split(
    descriptors_nf, targets_nf, test_size=0.2, stratify=targets_nf, random_state=random_state)

"""Models' list"""
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC(probability=True),
    'KNeighborsClassifier': KNeighborsClassifier()
}

"""Function for model evaluation"""


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


"""Evaluation"""
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train_hemo, X_test_hemo, y_train_hemo, y_test_hemo)

"""Results"""
print('HEMOLYSIS DATASET')
for name, metrics in results.items():
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name}: {value:.4f}")
    print("\n")

"""Evaluation"""
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train_nf, X_test_nf, y_train_nf, y_test_nf)

"""Results"""
print('NONFOULING DATASET')
for name, metrics in results.items():
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name}: {value:.4f}")
    print("\n")
