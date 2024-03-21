import pandas as pd
from app.utils.conctants import monomer_smiles
from app.sequant_tools import SequantTools
from app.utils.predict_utils import LazyClass_vae


initial_df = pd.read_csv('../utils/data/antiox.csv', encoding='cp1251')

# Retrieving only the required columns
needed_columns = ['Sequence', 'FRS']
antiox_df = initial_df.loc[:, needed_columns]
antiox_df.columns = ['seq', 'Label']

# Dropping duplicates in seq column
antiox_df = antiox_df.drop_duplicates(subset=['seq'])
antiox_df.reset_index(drop=True, inplace=True)

# Filtration based on max length = 96
antiox_df = antiox_df[antiox_df['seq'].apply(lambda x: len(x)) <= 96]

# Filtration based on known monomers
known_monomers = set(''.join(monomer_smiles.keys()))
filtered_df = antiox_df[~antiox_df['seq'].apply(lambda x: any(monomer not in known_monomers for monomer in x))]

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

model = LazyClass_vae(
    features=descriptors,
    target=targets
)

print(model)
