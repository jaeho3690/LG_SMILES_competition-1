from pathlib import Path

data_dir = Path('/home/jaeho_ubuntu/SMILES/data/')
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'
train_csv_dir = data_dir /'train.csv'
train_pickle_dir = data_dir /'train_modified.pkl'

input_data_dir = data_dir / 'input_data'

random_seed = 123