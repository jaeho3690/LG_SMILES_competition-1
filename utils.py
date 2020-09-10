import os
import csv

def _csv_writer(file_name, write_data):
    f = open(file_name, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(write_data)
    f.close()


def logger(log_data):
    _csv_writer('log.csv', log_data)


def make_directory(path):
    try:
        os.mkdir(path)
        print(path + ' is generated!')
    except OSError:
        pass

def load_reversed_token_map(path):
    """Gets the path of the reversed token map json"""
    with open(path, 'r') as j:
        reversed_token_map = json.load(j)
    return reversed_token_map

def decode_predicted_sequences(predicted_sequence_list,reversed_token_map):
    """
    Args:
        predicted_sequence_list: List of sequences in predicted form ex) [27,1,2,5]
        reveresed_token_map: Dictionary mapping of reversed token map
    Return:
        predicted_sequence_str: 
    """
    predicted_sequence_str = ""
    for e in predicted_sequence_list:
        if reversed_token_map[e]=='<unk>':
            continue
        elif reversed_token_map[e] in {'<end>','<pad>'}:
            break
        else:
            predicted_sequence_str+=reversed_token_map[e]
    
    return predicted_sequence_list



