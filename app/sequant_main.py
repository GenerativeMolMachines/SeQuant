import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv
import json
import numpy as np
import joblib
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import tensorflow as tf
from keras.models import Model
from sq_dataclasses import *

load_dotenv()


class SeqQuantKernel:
    """
    Class designed to process DNA/RNA/protein sequences with custom encoder using rdkit and peptide descriptors.
    """

    def __init__(
        self,
        polymer_type: str = 'protein',
        encoding_strategy='protein',
        new_monomers: NewMonomers = None,
    ):
        """
        Initialisation.
        :param polymer_type: Polymers types. Possible values: 'protein', 'DNA', 'RNA'.
        :param encoding_strategy: Selects a model for encoding. Possible values: 'protein', 'aptamers', 'nucleic_acids'.
        :param skip_unprocessable: Set to True to skip sequences with unknown monomers and sequences with length >96.
        :param new_monomers: list of dicts with new monomers: {'name':'str', 'class':'protein/DNA/RNA', 'smiles':'str'}
        """
        self.max_sequence_length: int = 96
        self.num_of_descriptors = 43
        self.descriptors_scaler = joblib.load(os.getenv('DESCRIPTORS_SCALER_PATH'))

        self.encoding_strategy = encoding_strategy
        self.model = self.load_model()

        self.polymer_type = polymer_type
        self.descriptors: dict[str, list[float]] = {}
        self.read_precalculated_rdkit_descriptors()
        self.new_monomers = new_monomers
        self.add_monomer_to_descriptors()
        self.known_monomers = set(self.descriptors.keys())

    def read_precalculated_rdkit_descriptors(self):
        """
        Formalizes the descriptors_file_path depending on the polymer type.
        """
        if self.polymer_type not in ['protein', 'DNA', 'RNA']:
            return ValueError(
                "Incorrect polymer_type. "
                "Use one from the list: 'protein', 'DNA', 'RNA'"
            )
        elif self.polymer_type == 'protein':
            descriptors_file_path = os.getenv('PROTEINS_DESCRIPTORS_PATH')
        elif self.polymer_type == 'DNA':
            descriptors_file_path = os.getenv('DNA_DESCRIPTORS_PATH')
        elif self.polymer_type == 'RNA':
            descriptors_file_path = os.getenv('RNA_DESCRIPTORS_PATH')
        with open(descriptors_file_path) as json_file:
            self.descriptors = json.load(json_file)

    def load_model(self):
        if self.encoding_strategy not in ['protein', 'aptamers', 'nucleic_acids']:
            return ValueError(
                "Incorrect type for encoding_strategy. "
                "Use one from the list: 'protein', 'aptamers', 'nucleic_acids'"
            )
        else:
            if self.encoding_strategy == 'protein':
                model_folder_path = os.getenv('PROTEINS_PATH')
            if self.encoding_strategy == 'aptamers':
                model_folder_path = os.getenv('APTAMERS_PATH')
            if self.encoding_strategy == 'nucleic_acids':
                model_folder_path = os.getenv('NUCLEIC_ACIDS_PATH')

        trained_model = tf.keras.models.load_model(model_folder_path)
        return Model(
            inputs=trained_model.input,
            outputs=trained_model.get_layer('Latent').output
        )

    def calculate_monomer(
            self,
            designation: str,
            smiles: str,
    ) -> dict:
        descriptor_names = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())

        get_descriptors = Chem.rdMolDescriptors.Properties(descriptor_names)
        molecule = Chem.MolFromSmiles(smiles)
        descriptors = np.array(
            get_descriptors.ComputeProperties(molecule)
        ).reshape((1, -1))
        descriptors_set = self.descriptors_scaler.transform(descriptors).tolist()[0]
        return {designation: descriptors_set}

    def add_monomer_to_descriptors(self):
        if self.new_monomers:
            for designation in self.new_monomers.monomers:
                if designation.name not in set(self.descriptors.keys()):
                    new_monomer = self.calculate_monomer(designation.name, designation.smiles)
                    self.descriptors.update(new_monomer)
                else:
                    error_text = f'Monomer "{designation}" is already exists in core.'
                    raise RuntimeError(error_text)

    def length_filter(self, sequence_list, skip_unprocessable):
        processed_sequence_list: list[str] = []
        for sequence in sequence_list:
            if len(sequence) > self.max_sequence_length:
                if not skip_unprocessable:
                    error_text = 'There are the sequence whose length ' \
                                 f'exceeds the maximum = {self.max_sequence_length}: {sequence} ' \
                                 'Set skip_unprocessable as True in kernel or exclude if by yourself.'
                    raise RuntimeError(error_text)
                else:
                    continue
            else:
                processed_sequence_list.append(sequence.upper())
        return processed_sequence_list

    def unknown_monomer_filter(self, sequence_list, skip_unprocessable):
        processed_sequence_list: list[str] = []
        for sequence in sequence_list:
            if not set(sequence).issubset(self.known_monomers):
                if not skip_unprocessable:
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

    def sequence_to_descriptor_matrix(
            self,
            sequence: str
    ) -> tf.Tensor:
        """
        Сonverts a single sequence into a descriptor matrix.
        :param sequence: Alphanumeric sequence.
        :return: Tensor with shape (max_sequence_length, num_of_descriptors).
        """
        sequence_matrix: tf.Tensor = tf.zeros(shape=[0, self.num_of_descriptors])
        for monomer in sequence:
            monomer_params = tf.constant(
                self.descriptors[monomer],
                dtype=tf.float32
            )
            descriptors_array = tf.expand_dims(
                monomer_params,
                axis=0
            )
            sequence_matrix = tf.concat(
                [sequence_matrix, descriptors_array],
                axis=0
            )
        sequence_matrix = tf.transpose(sequence_matrix)
        paddings = tf.constant([[0, 0], [0, 96 - len(sequence)]])
        sequence_matrix = tf.pad(
            sequence_matrix,
            paddings=paddings,
            mode='CONSTANT',
            constant_values=-1
        )
        return sequence_matrix

    def encoding(
            self,
            sequence_list
    ) -> tf.Tensor:
        """
        Сonverts a list of sequences into a  sequences/descriptor tensor.
        :return: Sequences/descriptor tensor.
        """
        container = []
        for sequence in sequence_list:
            seq_matrix = tf.expand_dims(
                self.sequence_to_descriptor_matrix(
                    sequence=sequence
                ),
                axis=0
            )
            container.append(seq_matrix)
        return tf.concat(container, axis=0)

    def generate_latent_representations(
            self,
            sequence_list,
            skip_unprocessable
    ) -> dict:
        """
        Processes the sequences/descriptor tensor using a model.
        :param sequence_list: Enumeration of sequences for filtering.
        :param skip_unprocessable: Set to True to skip sequences with unknown monomers and sequences with length >96.
        :return: Ready-made features.
        """
        result:dict[str, list[float]] = {}
        sequence_list_filter1 = self.length_filter(sequence_list, skip_unprocessable)
        sequence_list_filter2 = self.unknown_monomer_filter(sequence_list_filter1, skip_unprocessable)
        print(sequence_list_filter2)
        if not sequence_list_filter2:
            return {}
        encoded_sequence_list = self.encoding(sequence_list_filter2)
        latent_representation: np.ndarray = self.model.predict(
            encoded_sequence_list
        )
        for sequence, descriptors in zip(sequence_list_filter2, latent_representation):
            result[sequence] = descriptors.tolist()
        return result


if __name__ == "__main__":
    new_mon = NewMonomers(monomers=[Monomer(name="XY", smiles="C(C(=O)O)N"),Monomer(name="T", smiles="C(C(=O)O)N")])
    sqt = SeqQuantKernel(
        polymer_type=PolymerType('protein'),
        encoding_strategy=EncodingStrategy('protein'),
        new_monomers=new_mon
    )
    seq_list = ['AtgcxAtgcxAtgcxAtgcxAtgcxAAtgcxAtgcxAtgcx', 'GC']
    result = sqt.generate_latent_representations(seq_list, True)
    print(result)
