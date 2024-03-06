import pandas as pd
from app.sequant_tools import SequantTools


polymer_type = 'DNA'
seq_list = ['Atgc', 'GC']
seq_df = pd.DataFrame()

sqt = SequantTools(
    sequences=seq_list,
    polymer_type=polymer_type,
    model_folder_path=r'../app/utils/models/proteins',
    add_peptide_descriptors=True
)
X_ref = sqt.generate_latent_representations()
print(X_ref)
