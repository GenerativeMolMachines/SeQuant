import os
PROJECT_PATH = os.getcwd()
NUCLEIC_ACIDS_PATH = os.path.join(PROJECT_PATH, 'app/utils/models/nucleic_acids/')
PROTEINS_PATH = os.path.join(PROJECT_PATH, 'app/utils/models/proteins/')

if __name__ == "__main__":
    envs = f'{PROJECT_PATH=}\n' \
           f'{NUCLEIC_ACIDS_PATH=}\n' \
           f'{PROTEINS_PATH=}\n'

    with open('.env', 'w') as f:
        f.write(envs)