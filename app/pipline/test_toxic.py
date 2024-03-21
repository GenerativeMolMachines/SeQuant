import pandas as pd
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools
from app.utils.predict_utils import LazyClass_vae

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score


initial_df = pd.read_csv('../utils/data/toxic.csv', encoding='cp1251')

# Retrieving only the required columns
needed_columns = ['seq', 'Label']
toxic_df = initial_df.loc[:, needed_columns]

# Dropping duplicates in seq column
toxic_df = toxic_df.drop_duplicates(subset=['seq'])
toxic_df.reset_index(drop=True, inplace=True)

# Filtration based on max length = 96
toxic_df = toxic_df[toxic_df['seq'].apply(lambda x: len(x)) <= 96]

# Encoding labels
replace_dict = {'toxic': 1, 'non-toxic': 0}
toxic_df['Label'].replace(replace_dict, inplace=True)

# Filtration based on known monomers
known_monomers = set(''.join(monomer_smiles.keys()))
filtered_df = toxic_df[~toxic_df['seq'].apply(lambda x: any(monomer not in known_monomers for monomer in x))]

# Retrieving descriptors by using Sequant
sequences = filtered_df['seq']
max_peptide_length = 96
polymer_type = 'protein'

sqt = SequantTools(
    sequences=sequences,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    model_folder_path=r'../utils/models/proteins'
)

descriptors = sqt.generate_latent_representations()
targets = filtered_df['Label']

"""
classifier = LazyClass_vae(
    features=descriptors,
    target=targets
)

print(classifier)
"""

# Comparison with RF and SVC models from the article
X_train, X_test, y_train, y_test = train_test_split(descriptors, targets, test_size=0.2, random_state=42)
"""
# RandomForestClassifier
rf_model = RandomForestClassifier(class_weight='balanced', max_features=15, n_estimators=400, random_state=11111)
rf_model.fit(X_train, y_train)

# SVC
svc_model = SVC(kernel='rbf', C=7, gamma=0.35, random_state=2222)
svc_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
svc_pred = svc_model.predict(X_test)

# Evaluation of models performance
rf_f1 = f1_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

svc_f1 = f1_score(y_test, svc_pred)
svc_recall = recall_score(y_test, svc_pred)
svc_auc = roc_auc_score(y_test, svc_pred)
svc_accuracy = accuracy_score(y_test, svc_pred)

print("RandomForestClassifier:")
print(f"F1 Score: {rf_f1}")
print(f"Recall: {rf_recall}")
print(f"AUC: {rf_auc}")
print(f"Accuracy: {rf_accuracy}")

print("\nSVC:")
print(f"F1 Score: {svc_f1}")
print(f"Recall: {svc_recall}")
print(f"AUC: {svc_auc}")
print(f"Accuracy: {svc_accuracy}")
"""

# Comparison with voting classifier from the article
# Creating ML model
svc_rbf = SVC(kernel='rbf', C=7, gamma=0.35, random_state=2222)
svc_poly = SVC(kernel='poly', C=0.001, gamma=0.2)
lsvc = LinearSVC(C=0.2)
rf = RandomForestClassifier(class_weight='balanced', max_features=15, n_estimators=400, random_state=11111)
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)

# Creating Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('svc_rbf', svc_rbf),
    ('svc_poly', svc_poly),
    ('lsvc', lsvc),
    ('rf', rf),
    ('nb', nb),
    ('knn', knn)
], voting='hard')

# Training
voting_clf.fit(X_train, y_train)

# Prediction
vc_pred = voting_clf.predict(X_test)

# Evaluation of models performance
vc_f1 = f1_score(y_test, vc_pred)
vc_recall = recall_score(y_test, vc_pred)
vc_auc = roc_auc_score(y_test, vc_pred)
vc_accuracy = accuracy_score(y_test, vc_pred)

print("Voting Classifier:")
print(f"F1 Score: {vc_f1}")
print(f"Recall: {vc_recall}")
print(f"AUC: {vc_auc}")
print(f"Accuracy: {vc_accuracy}")
