import argparse

from utils import train_validation_split_df,create_input_files,str2bool
from config import data_dir,train_dir,test_dir,train_csv_dir,train_pickle_dir,sample_submission_dir,input_data_dir,random_seed

def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--split', default=False, type=str2bool,
                        help='Should we make a split to dataframe?')
    config = parser.parse_args()

    return config
if __name__ == '__main__':
    # Get configuration
    config = vars(parse_args())
    if config['split']==True:
        print('Carrying out split')
        train_validation_split_df(data_dir=data_dir,
                            train_csv_dir=train_csv_dir,
                            random_seed = random_seed,
                            train_size=0.8)

    # Create input files (along with word map)
    create_input_files(train_dir=train_dir,
                       train_pickle_dir=train_pickle_dir,
                       output_folder=input_data_dir,
                       test_dir = test_dir,
                       sample_submission_csv = sample_submission_dir,
                       min_token_freq=5,
                       max_len=75,
                       random_seed=random_seed)


