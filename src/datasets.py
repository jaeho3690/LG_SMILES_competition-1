import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class SmilesDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, base_file_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param base_file_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(data_folder/ f"{self.split}_IMAGES_{base_file_name}.hdf5", 'r')
        self.imgs = self.h['images']
        # Load encoded captions (completely into memory)
        with open(data_folder/ f"{self.split}_SMILES_SEQUENCES_{base_file_name}.json", 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(data_folder/ f"{self.split}_SMILES_SEQUENCE_LENS_{base_file_name}.json", 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # We need to figure out if there is more to return depending on the score type in Validation,  Test
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
