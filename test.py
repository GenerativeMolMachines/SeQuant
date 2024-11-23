import pandas as pd
from app.sequant_tools import SequantTools


max_peptide_length = 96
polymer_type = 'protein'
seq_list = ['ALKMNRRA', 'LLLLLL']
seq_df = pd.DataFrame()

sqt = SequantTools(
    sequences=seq_list,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    encoding_strategy='protein',
    ignore_unknown_monomer=True
)
X_ref = sqt.generate_latent_representations()
print(X_ref)
