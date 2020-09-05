import os
import argparse
import torch
import torchvision.transforms as transforms

from model.Model import MSTS
from src.datasets import SmilesDataset
from src.config import input_data_dir, base_file_name


parser = argparse.ArgumentParser()
# parser.add_argument('--data_folder', type=str, default='home/jaeho_ubuntu/SMILES/data/input_data', help='folder with image data files saved')
# parser.add_argument('--data_name', type=str, default='seed_123_max75smiles', help='csv file that contain information about image data')

parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings')
parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN')
parser.add_argument('--dropout', type=float, default=0.5, help='droup out rate' )
parser.add_argument('--device', type=str, default='cuda', help='sets device for model and PyTorch tensors')
parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='set to true only if inputs to model are fixed size; otherwise lot of computational overhead')

parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--epochs_since_improvement', type=int, default=0, help="keeps track of number of epochs since there's been an improvement in validation BLEU")
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--workers', type=int, default=1, help='for data-loading; right now, only 1 works with h5py')
parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning')
parser.add_argument('--decoder_lr', type=float, default=4e-4, help='learning rate for decoer')
parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of')
parser.add_argument('--alpha_c', type=float, default=1., help="regularization parameter for 'doubly stochastic attention', as in the paper")
parser.add_argument('--best_bleu4', type=float, default=0., help='BLEU-4 score right now')
parser.add_argument('--print_freq', type=int, default=100, help='print training/balidation stats every __ batches')
parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='fine-tune encoder')
parser.add_argument('--check_point', type=str, default=None, help='path to checkpoint, None if none')

config = parser.parse_args()

model = MSTS(config)


# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    SmilesDataset(input_data_dir, base_file_name, 'TRAIN', transform=transforms.Compose([normalize])),
    batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    SmilesDataset(input_data_dir, base_file_name, 'VAL', transform=transforms.Compose([normalize])),
    batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)


model.train(train_loader)




