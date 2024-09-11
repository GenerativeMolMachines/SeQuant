import pandas as pd

import sys
sys.path.insert(0, '/nfs/home/enam/SeQuant')

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
print("imports done")

random_state = 2024

# Data import
hemo_label = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/hemo_df_cut.csv') # seqs and labels
hemo_descriptors = pd.read_csv('/nfs/home/enam/SeQuant/app/utils/data/hemo_df_encoding.csv') # seqs and descriptors

# Data preprocessing
hemo_df = pd.merge(hemo_label, hemo_descriptors, on='seq', how='left')

print("Data has been imported")
print("\n")

# Predictions
targets_hemo = hemo_df['label']
descriptors_hemo = hemo_df.drop(columns=['seq', 'label'])
scaler = MinMaxScaler(feature_range=(-1, 1))
descriptor_hemo = scaler.fit_transform(descriptors_hemo)
print('Descriptors have been read')
print("\n")

# Stratified split
X_train_hemo, X_test_hemo, y_train_hemo, y_test_hemo = train_test_split(
    descriptor_hemo, targets_hemo, test_size=0.2, stratify=targets_hemo, random_state=random_state)

print("Sets have been split")
print("\n")

# Models' list
models = {
    'XGB': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(),
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


# Evaluation on test set
final_results_hemo = {}
for name, model in models.items():
    final_results_hemo[name] = evaluate_model(
        model, X_train_hemo, X_test_hemo, y_train_hemo, y_test_hemo
    )

# Print the results on test set
print('EVALUATION ON TEST SET (HEMOLYSIS DATASET)')
for name, metrics in final_results_hemo.items():
    print(f"Results for {name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name}: {value:.4f}")
    print("\n")
