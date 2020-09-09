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