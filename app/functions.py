import joblib
import numpy as np
import pandas as pd

import peptides
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def define_peptide_generated_descriptors(
        sequence: list,
) -> dict:
    """
    Generates an array of descriptors using the peptides lib.
    :return: Peptide descriptors
    """
    return {sequence:peptides.Peptide(sequence).descriptors()}


def calculate_monomer(
        designation: str,
        smiles: str,
        polymer_type: str
) -> dict:
    descriptors_set: list = []
    prefix: str = ''
    if len(designation) != 1:
        descriptor_names = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())

        get_descriptors = Chem.rdMolDescriptors.Properties(descriptor_names)
        molecule = Chem.MolFromSmiles(smiles)
        descriptors = np.array(
                get_descriptors.ComputeProperties(molecule)
            ).reshape((1, -1))
        scaler = joblib.load(r"app/utils/descriptors_scaler.pkl")
        descriptors_set = scaler.transform(descriptors).tolist()[0]
        if polymer_type == 'RNA':
            prefix = 'r'
        elif polymer_type == 'DNA':
            prefix = 'd'
    return {prefix + designation: descriptors_set}
