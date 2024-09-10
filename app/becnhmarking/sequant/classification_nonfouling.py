import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/enam/SeQuant')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
print("imports done")

random_state = 2024

# Data import
nf_label = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/nf_df.csv') # seqs and labels
nf_descriptors = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/nf_df_encoding.csv') # seqs and descriptors

# Data preprocessing
nf_df = pd.merge(nf_label, nf_descriptors, on='seq', how='left')

print("Data has been imported")
print("\n")

# Predictions
targets_nf = nf_df['label']
descriptors_nf = nf_df.drop(columns=['seq', 'label'])
scaler = MinMaxScaler()
descriptor_nf = scaler.fit_transform(descriptors_nf)
print('Descriptors have been read')
print("\n")

# Stratified split
X_train_nf, X_test_nf, y_train_nf, y_test_nf = train_test_split(
    descriptor_nf, targets_nf, test_size=0.2, stratify=targets_nf, random_state=random_state)

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


# Final evaluation on test set
final_results_nf = {}
for name, model in models.items():
    final_results_nf[name] = evaluate_model(
        model, X_train_nf, X_test_nf, y_train_nf, y_test_nf
    )

# Print evaluation on test set
print('EVALUATION ON TEST SET (NONFOULING DATASET)')
for name, metrics in final_results_nf.items():
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name}: {value:.4f}")
    print("\n")
