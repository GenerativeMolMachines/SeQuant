import numpy as np
import pandas as pd

import peptides


def define_peptide_generated_descriptors(
        self,
) -> np.ndarray:
    """
    Generates an array of descriptors using the peptides lib.
    :return: Peptide descriptors
    """
    peptide_descriptors = pd.DataFrame(
        [peptides.Peptide(seq).descriptors() for seq in self.filtered_sequences]
    )
    self.peptide_descriptor_names = list(peptide_descriptors.columns)
    self.peptide_descriptors = np.array(peptide_descriptors)

    if self.normalize:
        self.peptide_descriptors = self.scaler.fit_transform(self.peptide_descriptors)

    return self.peptide_descriptors