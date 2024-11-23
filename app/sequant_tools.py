import os
from dotenv import load_dotenv
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
        sequences: list[str] = [],
        polymer_type: str = '',
        encoding_strategy='',
        max_sequence_length: int = 96,
        normalize: bool = True,
        feature_range: tuple[int, int] = (-1, 1),
        add_peptide_descriptors: bool = False,
        new_monomers: list[dict] = [],
        ignore_unknown_monomer: bool = False
    ):
        """
        Initialisation.
        :param sequences: Enumeration of sequences for filtering.
        :param polymer_type: Polymers types. Possible values: 'protein', 'DNA', 'RNA'.
        :param encoding_strategy: Selects a model for encoding. Possible values: 'protein', 'aptamers', 'nucleic_acids'.
        :param max_sequence_length: The maximum number of characters in a sequence.
        :param normalize: Set to True to transform values with MinMaxScaler.
        :param feature_range: Desired range of transformed data.
        :param add_peptide_descriptors: Set to True to add peptide descriptors.
        :param ignore_unknown_monomer: Set to True to ignore unknown monomers.
        :param new_monomers: list of dicts with new monomers: {'name':'str', 'class':'protein/DNA/RNA', 'smiles':'str'}
        """
        self.descriptors: pd.DataFrame = pd.DataFrame()
        self.latent_representation_df = pd.DataFrame()
        self.filtered_sequences: list[str] = []
        self.prefix: str = ''
        self.encoded_sequences: tf.Tensor = []
        self.model: Model = None    # intermediate_layer_model
        self.peptide_descriptor_names: list[str] = []
        self.peptide_descriptors: npt.NDArray = []
        self.latent_representation: npt.NDArray = []
        self.descriptor_names: list[str] = []
        self.energy_names: list[str] = []

        self.sequences = sequences
        self.polymer_type = polymer_type
        self.max_length = max_sequence_length
        self.normalize = normalize
        self.feature_range = feature_range
        self.add_peptide_descriptors = add_peptide_descriptors
        self.new_monomers = new_monomers
        self.ignore_unknown_monomer = ignore_unknown_monomer
        self.encoding_strategy = encoding_strategy

        self.monomer_smiles_info: dict[str, str] = monomer_smiles
        self.add_monomers()
        self.scaler = MinMaxScaler(feature_range=self.feature_range)

        self.generate_rdkit_descriptors()
        self.filter_sequences()
        self.define_prefix()
        self.model_import()

    def generate_rdkit_descriptors(
            self
    ):
        """
        Generates descriptors for monomers in dict[monomer_name, smiles] using rdkit and DFT data.
        """
        self.descriptor_names = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())
        num_descriptors: int = len(self.descriptor_names)
        descriptors_set: npt.NDArray = np.empty((0, num_descriptors), float)

        get_descriptors = Chem.rdMolDescriptors.Properties(self.descriptor_names)

        for _, value in self.monomer_smiles_info.items():
            molecule = Chem.MolFromSmiles(value)
            descriptors = np.array(
                get_descriptors.ComputeProperties(molecule)
            ).reshape((-1, num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)

        descriptors_set = MinMaxScaler(feature_range=(-1, 1)).fit_transform(descriptors_set)

        descriptors_rdkit = pd.DataFrame(
            descriptors_set,
            columns=self.descriptor_names,
            index=list(self.monomer_smiles_info.keys())
        )

        energy_data = pd.read_csv('app/utils/data/energy_data.csv')
        energy_set = energy_data.set_index("Aminoacid").iloc[:, :]

        self.energy_names = list(energy_set.columns)

        scaled_energy = MinMaxScaler(feature_range=(-1, 1)).fit_transform(energy_set)
        scaled_energy_set = pd.DataFrame(
            scaled_energy,
            columns=self.energy_names,
            index=list(self.monomer_smiles_info.keys())
        )

        self.descriptors = pd.concat([descriptors_rdkit, scaled_energy_set], axis=1)

    def filter_sequences(
            self
    ):
        """
        Filters sequences based on the maximum length and content of known monomers.
        """
        all_sequences: list[str] = []
        bigger_then_max_sequence: list[str] = []
        for sequence in self.sequences:
            if len(sequence) <= self.max_length:
                all_sequences.append(sequence.upper())
            else:
                bigger_then_max_sequence.append(sequence)

        if len(bigger_then_max_sequence) != 0:
            error_text = f'There are the sequences whose length exceeds the maximum = {self.max_length}: \n' \
                        f'{bigger_then_max_sequence}.'
            raise ValueError(error_text)

        unknown_monomers_sequence = []
        known_monomers = set(self.monomer_smiles_info.keys())
        for sequence in all_sequences:
            if set(sequence).issubset(known_monomers):
                self.filtered_sequences.append(sequence)
            else:
                unknown_monomers_sequence.append(sequence)

        if not self.ignore_unknown_monomer and len(unknown_monomers_sequence) != 0:
            unknown_monomers: list[str] = []
            for sequence in unknown_monomers_sequence:
                unknown_monomers.extend(list(set(sequence) - known_monomers))

            unknown_monomers = list(set(unknown_monomers))
            error_text = 'There are unknown monomers in sequences: \n' \
                f'{unknown_monomers}. \n' \
                'Please add them in with using new_monomers parameter or set ignore_unknown_monomer as True.'
            raise ValueError(error_text)

    def define_prefix(self):
        """
        Formalizes the prefix depending on the polymer type.
        """
        assert self.polymer_type in ['protein', 'DNA', 'RNA'], "Possible values: 'protein', 'DNA', 'RNA'.\n"
        if self.polymer_type == 'protein':
            self.prefix = ''
        elif self.polymer_type == 'DNA':
            self.prefix = 'd'
        elif self.polymer_type == 'RNA':
            self.prefix = 'r'

    def model_import(self):
        """
        Initialise model
        """
        if self.encoding_strategy not in ['protein', 'aptamers', 'nucleic_acids']:
            return ValueError(
                "Incorrect type for encoding_strategy. Use one from the list: 'protein', 'aptamers', 'nucleic_acids'"
            )
        else:
            if self.encoding_strategy == 'protein':
                self.model_folder_path = os.getenv('PROTEINS_PATH')
            if self.encoding_strategy == 'aptamers':
                self.model_folder_path = os.getenv('APTAMERS_PATH')
            if self.encoding_strategy == 'nucleic_acids':
                self.model_folder_path = os.getenv('NUCLEIC_ACIDS_PATH')

        trained_model = tf.keras.models.load_model(self.model_folder_path)
        layer_name = 'Latent'
        self.model = Model(
            inputs=trained_model.input,
            outputs=trained_model.get_layer(layer_name).output
        )

    def sequence_to_descriptor_matrix(
            self,
            sequence: str
    ) -> tf.Tensor:
        """
        Сonverts a single sequence into a descriptor matrix.
        :param sequence: Alphanumeric sequence.
        :return: Tensor with shape (max_sequence_length, num_of_descriptors).
        """
        rows: int = self.descriptors.shape[1]
        sequence_matrix: tf.Tensor = tf.zeros(shape=[0, rows])  # shape (0,rows)
        for monomer in sequence:
            monomer_params = tf.constant(
                self.descriptors.loc[self.prefix + monomer],
                dtype=tf.float32
            )
            descriptors_array = tf.expand_dims(
                monomer_params,
                axis=0  # shape (1,rows)
            )
            sequence_matrix = tf.concat(
                [sequence_matrix, descriptors_array],
                axis=0
            )
        sequence_matrix = tf.transpose(sequence_matrix)
        shape = sequence_matrix.get_shape().as_list()[1]
        if shape < 96:
            paddings = tf.constant([[0, 0], [0, 96 - shape]])
            sequence_matrix = tf.pad(
                sequence_matrix,
                paddings=paddings,
                mode='CONSTANT',
                constant_values=-1
            )
        return sequence_matrix

    def encoding(
            self
    ) -> tf.Tensor:
        """
        Сonverts a list of sequences into a  sequences/descriptor tensor.
        :return: Sequences/descriptor tensor.
        """
        container = []
        for sequence in tqdm(self.filtered_sequences):
            seq_matrix = tf.expand_dims(
                self.sequence_to_descriptor_matrix(
                    sequence=sequence
                ),
                axis=0
            )
            container.append(seq_matrix)

        self.encoded_sequences = tf.concat(container, axis=0)
        return self.encoded_sequences

    def generate_latent_representations(
            self,
            dataframe=True
    ):
        """
        Processes the sequences/descriptor tensor using a model.
        :return: Ready-made features.
        """
        self.encoding()
        self.latent_representation: np.ndarray = self.model.predict(
            self.encoded_sequences
        )

        if self.add_peptide_descriptors:
            self.define_peptide_generated_descriptors()
            self.latent_representation = np.concatenate(
                (self.latent_representation, self.peptide_descriptors),
                axis=1
            )

        if dataframe:
            descriptor_names_repl = [
                i + '_repr' for i in self.descriptor_names + self.peptide_descriptor_names + self.energy_names
            ]
            self.latent_representation_df = pd.DataFrame(
                self.latent_representation,
                columns=descriptor_names_repl,
                index=self.filtered_sequences
            )
            return self.latent_representation_df

        return self.latent_representation

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

    def add_monomers(self):
        """
        Adds new monomers to the monomer_smiles_info: dict[str, str]
        """
        if not self.ignore_unknown_monomer:
            for item in self.new_monomers:
                name = item['name']
                prefix = ''
                if item['class'] == 'RNA':
                    prefix = 'r'
                elif item['class'] == 'DNA':
                    prefix = 'd'
                name = prefix + name
                if name not in self.monomer_smiles_info:
                    self.monomer_smiles_info[name] = item['smiles']
