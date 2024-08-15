import os
PROJECT_PATH = os.getcwd()
APTAMERS_PATH = os.path.join(PROJECT_PATH, 'app/utils/models/aptamers/')
NUCLEIC_ACIDS_PATH = os.path.join(PROJECT_PATH, 'app/utils/models/nucleic_acids/')
PROTEINS_PATH = os.path.join(PROJECT_PATH, 'app/utils/models/proteins/')
PROTEINS_DESCRIPTORS_PATH = os.path.join(PROJECT_PATH, 'app/utils/descriptors/protein_descriptors.json')
RNA_DESCRIPTORS_PATH = os.path.join(PROJECT_PATH, 'app/utils/descriptors/rna_descriptors.json')
DNA_DESCRIPTORS_PATH = os.path.join(PROJECT_PATH, 'app/utils/descriptors/dna_descriptors.json')
DESCRIPTORS_SCALER_PATH = os.path.join(PROJECT_PATH, 'app/utils/models/descriptors_scaler.pkl')

if __name__ == "__main__":
    envs = f'{PROJECT_PATH=}\n' \
           f'{APTAMERS_PATH=}\n' \
           f'{NUCLEIC_ACIDS_PATH=}\n' \
           f'{PROTEINS_PATH=}\n' \
           f'{PROTEINS_DESCRIPTORS_PATH=}\n' \
           f'{RNA_DESCRIPTORS_PATH=}\n' \
           f'{DNA_DESCRIPTORS_PATH=}\n' \
           f'{DESCRIPTORS_SCALER_PATH=}\n'

    with open('.env', 'w') as f:
        f.write(envs)