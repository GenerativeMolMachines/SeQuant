import pandas as pd
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools
from app.utils.predict_utils import LazyClass_vae


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

classifier = LazyClass_vae(
    features=descriptors,
    target=targets
)

print(classifier)
