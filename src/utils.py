import os
import sys
import numpy as np
import pandas as pd
import h5py
import json
from tqdm import tqdm
from collections import Counter
import argparse
import warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 80

from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg


def train_test_split_df(data_dir,train_csv_dir,random_seed,train_size=0.8,val_size=0.25):
    """ Split images into train,test,val in the dataframe and save them to pickle
    Args:
        data_dir: Data directory
        train_csv_dir: Directory of train csv directory
    
    """
    df = pd.read_csv(train_csv_dir)

    # Create Necessary Columns
    df['SMILES_TOKEN']= df['SMILES'].apply(lambda x: split_to_token(x))
    df['LEN_SMILES']= df['SMILES'].apply(lambda x:len(x))
    df.groupby('LEN_SMILES').count().to_csv('/home/jaeho_ubuntu/SMILES/data/count.csv')
    df_shuffled = df.sample(frac=1,random_state= random_seed).reset_index(drop=True)
    
    del df

    train_index = int(len(df_shuffled)*train_size)
    val_index = int(train_index*val_size)
    df_train = df_shuffled.iloc[:train_index,:]
    df_val = df_shuffled.iloc[train_index-val_index:train_index,:]
    df_test = df_shuffled.iloc[train_index:,:]

    print(f'Training: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}')

    # Set Column Value for split
    df_train.loc[:,'split']='train'
    df_val.loc[:,'split']='val'
    df_test.loc[:,'split']='test'

    df = pd.concat([df_train,df_val,df_test],axis=0)

    del df_train,df_val,df_test

    # Save to pickle file 
    df.to_pickle( data_dir /'train_modified.pkl')
    print(f"Saved as 'train_modified.pkl'")


def create_input_files(train_dir,train_pickle_dir,output_folder,min_token_freq,max_len=75,random_seed=123):
    """ Creates input files for train, val, test data
    Args:
        train_dir: Directory of train image folder
        train_csv_dir: Directory of train csv directory
        output_folder: Directory to save files
        min_token_freq: token that occurs less frequently than this threshold are binned as <unk>s
        max_len: Maximum Length of smiles_caption. The maximum length of smiles_caption
    """
    df = pd.read_pickle(train_pickle_dir)

    df = df.sample(n=10000)
    df.reset_index(inplace=True)
    print(df.head())
    # Read image paths and SMILES for each image
    train_image_paths = []
    train_image_smiles = []
    val_image_paths = []
    val_image_smiles = []
    test_image_paths = []
    test_image_smiles = []
    token_freq= Counter()


    for index in tqdm(df.index,desc='Looping index'):
        try:
            smiles_caption=[]
            for token in df.loc[index,'SMILES_TOKEN']:
                # Update token frequency
                token_freq.update(token)
            
            if len(df.loc[index,'SMILES'])==0:
                continue

            smiles_caption.append(df.loc[index,'SMILES'])

            path = train_dir / df.loc[index,'file_name']


            split_location = df.loc[index,'split']

            if  split_location in {'train'}:
                train_image_paths.append(path)
                train_image_smiles.append(smiles_caption)
            elif split_location in {'val'}:
                val_image_paths.append(path)
                val_image_smiles.append(smiles_caption)
            elif split_location in {'test'}:
                test_image_paths.append(path)
                test_image_smiles.append(smiles_caption)
        except:
            continue

    # Sanity Check
    assert len(train_image_paths) == len(train_image_smiles)
    assert len(val_image_paths) == len(val_image_smiles)
    assert len(test_image_paths) == len(test_image_smiles)

    # Create token map
    tokens = [t for t in token_freq.keys() if token_freq[t] > min_token_freq]
    token_map = {k: v + 1 for v, k in enumerate(tokens)}
    token_map['<unk>'] = len(token_map) + 1
    token_map['<start>'] = len(token_map) + 1
    token_map['<end>'] = len(token_map) + 1
    token_map['<pad>'] = 0

    print(token_map)
    base_filename = f'seed_{random_seed}_max{max_len}smiles'

    with open(output_folder / f'TOKENMAP_{base_filename}.json','w') as j:
        json.dump(token_map,j)
        print(f'Saved TOKENMAP_{base_filename}.json')
    
    for impaths, smiles_captions, split in [(train_image_paths, train_image_smiles, 'TRAIN'),
                                   (val_image_paths, val_image_smiles, 'VAL'),
                                   (test_image_paths, test_image_smiles, 'TEST')]:

        with h5py.File(output_folder / f"{split}_IMAGES_{base_filename}.hdf5", 'w') as h:
            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print(f"\nReading {split} images and captions, storing to file...\n")

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):
                try:
                    # Read images
                    img = Image.open(impaths[i])
                    img= img.resize((256,256))
                    img = np.array(img)
                    img = np.rollaxis(img, 2, 0)
                    assert img.shape == (3, 256, 256)
                    assert np.max(img) <= 255

                    # Save image to HDF5 file
                    images[i] = img

                    smiles_caption = smiles_captions[i]
                    for j, c in enumerate(smiles_caption):

                        # Encode captions
                        enc_c = [token_map['<start>']] + [token_map.get(token, token_map['<unk>']) for token in c] + [
                            token_map['<end>']] + [token_map['<pad>']] * (max_len - len(c))
                        # Find caption lengths
                        c_len = len(c) + 2

                        enc_captions.append(enc_c)
                        caplens.append(c_len)
                except:
                    continue
            # Sanity check
            assert images.shape[0] == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(output_folder/ f'{split}_SMILES_CAPTIONS_{base_filename}.json', 'w') as j:
                json.dump(enc_captions, j)

            with open(output_folder/ f'{split}_SMILES_CAPLENS_{base_filename}.json', 'w') as j:
                json.dump(caplens, j)




def split_to_token(word,window=1):
    """ Split string into tokens with given window size
    """
    chunk_size = len(word)-(window-1)
    return [word[i:i+window] for i in range(0,chunk_size)]


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')