# LG_SMILES_competition

## Requirements
### conda lib
### python lib

## Data generation

## How to train
### data preprocessing
```

```

### training
You will need to modify the src/config.py to accustom your directory setting. Running below scripts will return a .hdf5, .json files needed for training and testing.
```
# Only if you need to make training and validation set 
python --split True --train_file True
# If you need to make a test set provided by LG
python --test_file True
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
