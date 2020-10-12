# LG_SMILES_competition

## Requirements
### conda lib
### python lib

## Data generation

## How to train
### data preprocessing
You will need to modify the src/config.py to accustom your directory setting.
```python
# The data_dir should point the data directory folder. All training and testing files should be placed below
data_dir = Path('/home/jaeho_ubuntu/SMILES/data/')
train_dir = data_dir / 'train' # all train_<number>.png files should be under the train folder
test_dir = data_dir / 'test'  # all test_<number>.png files should be under the test folder
train_csv_dir = data_dir /'train.csv'  #train.csv file provided by LG 
# Sample submission directory
sample_submission_dir = data_dir /'sample_submission.csv'

# train_modified contains a modifed version of train.csv. 
# Information of the train/validation split is stored here. I saved in pickle just for efficiency
train_pickle_dir = data_dir /'train_modified.pkl'

### Data directory containing .hdf5, .json file files.
input_data_dir = data_dir / 'input_data'
base_file_name = 'seed_123_max75smiles'

### seed for train/val split
random_seed = 123

### Reversed_token_file used to map numbers to string. 
reversed_token_map_dir = input_data_dir/ f'REVERSED_TOKENMAP_{base_file_name}.json'
```

Running below scripts will return a .hdf5, .json files needed for training and testing.
```
# Only if you need to make training and validation set 
python --split True --train_file True
# If you need to make a test set provided by LG
python --test_file True
```

### training

```

```

## How to test

### simgle model test
```
python main.py --work_type single_test --model_load_path <path where the model is saved> --model_load_num <model number> 
```

### ensemble test

```
python main.py --work_type ensemble_test --model_load_path <path where the model is saved>
```
