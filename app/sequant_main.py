import os
from dotenv import load_dotenv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.typing as npt

import peptides
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import tensorflow as tf
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from .utils.conctants import monomer_smiles
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SequantTools:
    """
    Class designed to process DNA/RNA/protein sequences with custom encoder using rdkit and peptide descriptors.
    """

    def __init__(
        self,
        polymer_type: str = '',
        encoding_strategy='protein',
        new_monomers: list[dict] = [],
        skip_unprocessable: bool = False
    ):
        """
        Initialisation.
        :param polymer_type: Polymers types. Possible values: 'protein', 'DNA', 'RNA'.
        :param encoding_strategy: Selects a model for encoding. Possible values: 'protein', 'aptamers', 'nucleic_acids'.
        :param skip_unprocessable:
        Set to True to skip sequences with unknown monomers and sequences with length >96.
        :param new_monomers: list of dicts with new monomers: {'name':'str', 'class':'protein/DNA/RNA', 'smiles':'str'}
        """
        self.polymer_type = polymer_type
        self.encoding_strategy = encoding_strategy
        self.new_monomers = new_monomers
        self.skip_unprocessable: bool = skip_unprocessable

        self.max_sequence_length: int = 96

        self.descriptors: dict[str, list[float]] = {}

    def read_precalculated_rdkit_descriptors(self):
        with open('data.json') as json_file:
            self.descriptors = json.load(json_file)

    def length_filter(self, sequence_list):
        processed_sequence_list: list[str] = []
        for sequence in sequence_list:
            if len(sequence) > self.max_sequence_length:
                if not self.skip_unprocessable:
                    error_text = 'There are the sequence whose length ' \
                                 f'exceeds the maximum = {self.max_sequence_length}: {sequence} ' \
                                 'Set skip_unprocessable as True in kernel or exclude if by yourself.'
                    raise RuntimeError(error_text)
                else:
                    continue
            else:
                processed_sequence_list.append(sequence.upper())
        return processed_sequence_list

    def unknown_monomer_filter(self, sequence_list):
        processed_sequence_list: list[str] = []
        known_monomers = set(self.descriptors.keys())
        for sequence in sequence_list:
            if not set(sequence).issubset(known_monomers):
                if not self.skip_unprocessable:
                    error_text = f'There are unknown monomers in sequence: {sequence}. ' \
                                'You can fix it with: \n' \
                                '1) adding new monomer in kernel;\n' \
                                '2) setting skip_unprocessable as True;\n' \
                                '3) excluding it by yourself.'
                    raise RuntimeError(error_text)
                else:
                    continue
            else:
                processed_sequence_list.append(sequence)
        return processed_sequence_list



