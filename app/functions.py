import numpy as np
import pandas as pd

import peptides


def define_peptide_generated_descriptors(
        sequences: list,
) -> dict:
    """
    Generates an array of descriptors using the peptides lib.
    :return: Peptide descriptors
    """
    return {seq:peptides.Peptide(seq).descriptors() for seq in sequences}
