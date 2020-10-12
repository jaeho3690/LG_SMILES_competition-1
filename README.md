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
When you try to testing our model by `--work_type single_test` or `--work_type ensemble_test`, you should set the flags.
### simgle model test
```
python main.py --work_type single_test --model_load_path <path where the model is saved> --model_load_num <model number> --test_file_path <path where the test images are saved>
```

### ensemble test

```
python main.py --work_type ensemble_test --model_load_path <path where the model is saved> --test_file_path <path where the test images are saved>
```


## Optional Arguments

| optional arguments | typs | default | help |
|---|:---:|:---:|:---|
|`--work_type` | str |  `train'` | choose work type 'train' or 'test' |
|`--encoder_type` | str |  `'wide_res'` | choose encoder model type 'wide_res', 'res', and 'resnext'  |
|`--seed` | int |  `1` | set the seed of model |
|`--decode_length` | int |  `140` | length of decoded SMILES sequence |
|`--emb_dim` | int |  `512` | dimension of word embeddings |
|`--attention_dim` | int |  `512` | dimension of attention linear layers |
|`--decoder_dim` | int |  `512` | dimension of decoder RNN |
|`--dropout` | float |  `0.5` | droup out rate |
|`--device` | str |  `'cuda'` | sets device for model and PyTorch tensors |
|`--gpu_non_block` | bool |  `True` | GPU non blocking flag |
|`--cudnn_benchmark` | bool |  `True` | set to true only if inputs to model are fixed size; otherwise lot of computational overhead |
|`--start_epoch` | int |  `0` | number of start epoch |
|`--epochs` | int |  `50` | number of epochs to train for |
|`--batch_size` | int |  `384` | batch size |
|`--workers` | int |  `8` | for data-loading; right now, only 1 works with h5py |
|`--encoder_lr` | float |  `1e-4` | learning rate for encoder if fine-tuning |
|`--decoder_lr` | float |  `4e-4` | learning rate for decoer |
|`--grad_clip` | float |  `5.` | clip gradients at an absolute value of |
|`--best_bleu4` | float |  `0.` | BLEU-4 score right now |
|`--fine_tune_encoder` | bool |  `True` | fine-tune encoder |
|`--model_save_path` | str |  `'graph_save'` | model save path |
|`--model_load_path` | str |  `None` | model load path |
|`--model_load_num` | int |  `None` | epoch number of saved model |
|`--test_file_path` | str |  `None` | test file path |