import pandas as pd
import numpy as np
import requests
import json
import time
import tensorflow as tf
import torch

from tqdm import tqdm
from transformers import logging
from transformers import TFAutoModel, AutoTokenizer
from sklearn.preprocessing import OneHotEncoder
from Bio.Align import substitution_matrices
from itertools import product
from peptides import Peptide


# This code is used to encode peptide sequences via different methods:
#   1) One-hot encoding
#   2) Blossum62 encoding
#   3) Threemers encoding
#   4) ProtBERT embeddings
#   5) SeQuant (API) embeddings


############################### Encoding functions ###############################

def one_hot_encode(sequence):
    encoder = OneHotEncoder(categories=[list(amino_acids)], dtype=int, sparse_output=False)
    sequence_array = np.array(list(sequence)).reshape(-1, 1)
    encoded = encoder.fit_transform(sequence_array).flatten()

    return encoded


def threemers_encode(sequence):
    k = 3
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    kmer_to_index = {kmer: idx for idx, kmer in enumerate([''.join(p) for p in product(amino_acids, repeat=k)])}
    encoded = [kmer_to_index[kmer] for kmer in kmers]

    return encoded


def blosum62_encode(sequence):
    blosum62 = substitution_matrices.load('BLOSUM62')
    encoded_vector = []

    for i in range(len(sequence) - 1):
        pair = (sequence[i], sequence[i+1])

        if pair in blosum62:
            encoded_vector.append(blosum62[pair])

        elif (pair[1], pair[0]) in blosum62:
            encoded_vector.append(blosum62[(pair[1], pair[0])])

        else:
            encoded_vector.append(0)

    return encoded_vector


def prot_bert_encode(sequence, model_name='Rostlab/prot_bert', device='GPU'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = TFAutoModel.from_pretrained(model_name, from_pt=True)

    inputs = tokenizer(sequence, return_tensors='tf', padding=False, truncation=False)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with tf.device(f'/{device}:0'):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, training=False)

    hidden_states = outputs.last_hidden_state

    embedding = tf.reduce_mean(hidden_states, axis=1).numpy().squeeze()

    return embedding


def process_dataset(df, encoding_func, pad_value):
    encoded_data = df['seq'].progress_apply(encoding_func)
    max_len = max(encoded_data.apply(len))

    encoded_data = encoded_data.apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant', constant_values=pad_value))

    encoded_df = pd.DataFrame(encoded_data.tolist(), index=df.index)

    result_df = pd.concat([df, encoded_df], axis=1)

    return result_df


############################### Encode sequences with functions ###############################


amino_acids = 'ACDEFGHIKLMNPQRSTVWYU'
datasets = ['antimic', 'antidia', 'antiinf', 'antiox', 'regression']

logging.set_verbosity_error()
tqdm.pandas()

for dataset in datasets:

    print(f'Start processing {dataset} dataset')

    df = pd.read_csv(f'data/{dataset}.csv')

    # One-hot encoding with padding 20
    one_hot_df = process_dataset(df, one_hot_encode, pad_value=0)
    one_hot_df.to_csv(f'data/encoded/{dataset}_one_hot.csv', index=False)

    # Threemers encoding with padding of max threemer index + 1
    threemers_pad_value = len(amino_acids) ** 3
    threemers_df = process_dataset(df, threemers_encode, pad_value=threemers_pad_value)
    threemers_df.to_csv(f'data/encoded/{dataset}_threemers.csv', index=False)

    # BLOSUM62 encoding with padding 0
    blosum62_df = process_dataset(df, blosum62_encode,  pad_value=0)
    blosum62_df.to_csv(f'data/encoded/{dataset}_blosum62.csv', index=False)

    # ProtBERT encoding with padding 0
    protbert_df = process_dataset(df, prot_bert_encode, pad_value=0)
    protbert_df.to_csv(f'data/encoded/{dataset}_protbert.csv', index=False)


############################### SeQuant API usage for SeQuant embeddings ###############################

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

for dataset in datasets:
    
    print(f'Start processing {dataset} dataset')

    df = pd.read_csv(f'data/{dataset}.csv')
    sequences = list(df['seq'])

    several_id_lists = np.array_split(np.asarray(sequences), int(len(sequences) / 50) + 1)

    df_fin_data = pd.DataFrame()

    for i in several_id_lists:
        params = {
            'sequences': ', '.join(list(i)),
            'polymer_type': 'protein',
            'encoding_strategy': 'protein',
            'skip_unprocessable': 'true',
        }

        time.sleep(1)
        response = requests.post('https://ai-chemistry.itmo.ru/api/encode_sequence', params=params, headers=headers)
        assert response.status_code == 200
        a = json.loads(response.content)
        data = pd.DataFrame.from_dict(a, orient='index')
        df_fin_data = pd.concat([df_fin_data, data])

    df_fin_data['seq'] = df_fin_data.index

    final_df = pd.merge(df, df_fin_data, on='seq', how='inner')

    final_df.to_csv(f'data/encoded/{dataset}_sequant.csv', index=False)


############################### Adding physicochemical properties to regression dataset  ###############################


# Getting physicochemical properties from Peptides package
def compute_properties(sequence):
    peptide = Peptide(sequence)
    return {
        'Instability_Index': peptide.instability_index(),
        'Theoretical_Net_Charge': peptide.charge(),
        'Isoelectric_Point': peptide.isoelectric_point(),
        'Molecular_Weight': peptide.molecular_weight()
    }


encodings = ['one_hot', 'threemers', 'blosum62', 'protbert', 'sequant']
targets = ['Instability_Index', 'Theoretical_Net_Charge', 'Isoelectric_Point', 'Molecular_Weight']

for encoding in encodings:

    print(f'Start processing {encoding} dataset')

    df = pd.read_csv(f'data/encoded/regression_{encoding}.csv')
    
    properties_test = df['seq'].progress_apply(compute_properties).apply(pd.Series)
    result_test = pd.concat([df, properties_test], axis=1)

    for target in targets:
        test_target = result_test.drop(columns=[t for t in targets if t != target])
        test_target = test_target.rename(columns={target: 'target'})

        test_target.to_csv(f'data/encoded/{target}_{encoding}.csv', index=False)

