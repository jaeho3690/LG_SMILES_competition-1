import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from PIL import Image

from rdkit import Chem
from rdkit import DataStructs

from model.Network import Encoder, DecoderWithAttention, PredictiveDecoder
from utils import make_directory, decode_predicted_sequences

import random
import numpy as np
import asyncio
import os


class MSTS:
    def __init__(self, config):
        # self._data_folder = config.data_folder
        # self._data_name = config.data_name
        self._work_type = config.work_type
        self._seed = config.seed

        self._vocab_size = 70
        self._decode_length = config.decode_length
        self._emb_dim = config.emb_dim
        self._attention_dim = config.attention_dim
        self._decoder_dim = config.decoder_dim
        self._dropout = config.dropout
        self._device = config.device
        self._cudnn_benchmark = config.cudnn_benchmark

        self._start_epoch = config.start_epoch
        self._epochs = config.epochs
        self._epochs_since_improvement = config.epochs_since_improvement
        self._batch_size = config.batch_size
        self._workers = config.workers
        self._encoder_lr = config.encoder_lr
        self._decoder_lr = config.decoder_lr
        self._grad_clip = config.grad_clip
        self._alpha_c = config.alpha_c
        self._best_bleu4 = config.best_bleu4
        self._print_freq = config.print_freq
        self._fine_tune_encoder = config.fine_tune_encoder

        self._model_save_path = config.model_save_path
        self._model_load_path = config.model_load_path
        self._model_load_num = config.model_load_num

        self._model_name = self._model_name_maker()

        self._seed_everything(self._seed)

        if self._work_type == 'train':
            self._decoder = DecoderWithAttention(attention_dim=self._attention_dim,
                                                 embed_dim=self._emb_dim,
                                                 decoder_dim=self._decoder_dim,
                                                 vocab_size=self._vocab_size,
                                                 dropout=self._dropout)
            self._decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                     self._decoder.parameters()),
                                                       lr=self._decoder_lr)
        elif self._work_type == 'test':
            self._decoder = PredictiveDecoder(attention_dim=self._attention_dim,
                                              embed_dim=self._emb_dim,
                                              decoder_dim=self._decoder_dim,
                                              vocab_size=self._vocab_size)

        self._encoder = Encoder()
        self._encoder.fine_tune(self._fine_tune_encoder)
        self._encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                 self._encoder.parameters()),
                                                   lr=self._encoder_lr) if self._fine_tune_encoder else None
        self._encoder.to(self._device)
        self._decoder.to(self._device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self._encoder = nn.DataParallel(self._encoder)
        #    self._decoder = nn.DataParallel(self._decoder)
        self._criterion = nn.CrossEntropyLoss().to(self._device)

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    def _clip_gradient(self, optimizer, grad_clip):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def train(self, train_loader):

        self._encoder.train()
        self._decoder.train()

        mean_loss = 0
        mean_accuracy = 0

        for i, (imgs, sequence, sequence_lens) in enumerate(train_loader):
            imgs = imgs.to(self._device)
            sequence = sequence.to(self._device)
            sequence_lens = sequence_lens.to(self._device)

            imgs = self._encoder(imgs)
            predictions, caps_sorted, decode_lengths, alphas, sort_ind = self._decoder(imgs, sequence, sequence_lens)

            targets = caps_sorted[:, 1:]

            # Calculate accuracy
            accr = self._accuracy_calcluator(predictions.detach().cpu().numpy(),
                                             targets.detach().cpu().numpy())
            mean_accuracy = mean_accuracy + (accr - mean_accuracy) / (i + 1)

            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = self._criterion(predictions, targets)
            mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (i + 1)

            # Back prop.
            self._decoder_optimizer.zero_grad()
            self._encoder_optimizer.zero_grad()

            loss.backward()

            # Clip gradients
            if self._grad_clip is not None:
                self._clip_gradient(self._decoder_optimizer, self._grad_clip)
                self._clip_gradient(self._encoder_optimizer, self._grad_clip)

            # Update weights
            self._decoder_optimizer.step()
            self._encoder_optimizer.step()

        return mean_loss, mean_accuracy

    def validation(self, val_loader):
        self._encoder.eval()
        self._decoder.eval()

        mean_loss = 0
        mean_accuracy = 0

        for i, (imgs, sequence, sequence_lens) in enumerate(val_loader):
            imgs = imgs.to(self._device)
            sequence = sequence.to(self._device)
            sequence_lens = sequence_lens.to(self._device)

            imgs = self._encoder(imgs)
            predictions, caps_sorted, decode_lengths, _, _ = self._decoder(imgs, sequence, sequence_lens)
            targets = caps_sorted[:, 1:]

            accr = self._accuracy_calcluator(predictions.detach().cpu().numpy(),
                                             targets.detach().cpu().numpy())

            mean_accuracy = mean_accuracy + (accr - mean_accuracy) / (i + 1)

            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = self._criterion(predictions, targets)
            mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (i + 1)
            del (loss, predictions, caps_sorted, decode_lengths, targets)

        return mean_loss, mean_accuracy

    def model_test(self, submission, data_list, reversed_token_map):

        self._encoder.eval()
        self._decoder.eval()

        is_smiles = False

        for i, img_path in enumerate(data_list):
            img = Image.open(img_path)
            imgs = self.png_to_tensor(img).to(self._device)

            predictions = None
            decoded_sequences = None
            is_smiles = False
            add_seed = 0

            while not is_smiles:
                self._seed_everything(self._seed + add_seed)
                imgs = self._encoder(imgs)
                predictions = self._decoder(imgs, self._decode_length)

                SMILES_predicted_sequence = list(torch.argmax(predictions.detach().cpu(), -1).numpy())[0]
                decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, reversed_token_map)

                is_smiles = self.is_smiles(decoded_sequences)
                add_seed += 1


            submission['SMILES'].loc[i] = decoded_sequences
            del (predictions)

        return submission

    def png_to_tensor(self, img: Image):
        img = img.resize((256,256))
        pixel = np.array(img)
        pixel = np.moveaxis(pixel, -1, 0)
        return torch.FloatTensor([pixel]) / 255.

    def is_smiles(self, sequence):
        try:
            Chem.MolFromSmiles(sequence)
        except:
            return False
        return True

    def model_save(self, save_num):
        torch.save(
            self._decoder.state_dict(),
            '../' + '{}/'.format(self._model_save_path) + self._model_name + '/decoder{}.pkl'.format(
                str(save_num).zfill(3))
        )
        torch.save(
            self._encoder.state_dict(),
            '../' + '{}/'.format(self._model_save_path) + self._model_name + '/encoder{}.pkl'.format(
                str(save_num).zfill(3))
        )

    def model_load(self):
        self._decoder.load_state_dict(
            torch.load('{}/decoder{}.pkl'.format(self._model_load_path, str(self._model_load_num).zfill(3)))
        )
        self._encoder.load_state_dict(
            torch.load('{}/encoder{}.pkl'.format(self._model_load_path, str(self._model_load_num).zfill(3)))
        )

    def _model_name_maker(self):
        name = 'model-emb_dim_{}-attention_dim_{}-decoder_dim_{}-dropout_{}-batch_size_{}'.format(
            self._emb_dim, self._attention_dim, self._decoder_dim, self._dropout, self._batch_size)
        make_directory(self._model_save_path + '/' + name)

        return name

    def _accuracy_calcluator(self, prediction: np.array, target: np.array):
        prediction = np.argmax(prediction, 2)
        l_p = prediction.shape[1]
        l_t = target.shape[1]
        dist = abs(l_p - l_t)

        if l_p > l_t:
            accr = np.array(prediction[:, :-dist] == target, dtype=np.int).mean()
        elif l_p < l_t:
            accr = np.array(prediction == target[:, :-dist], dtype=np.int).mean()
        else:
            accr = np.array(prediction == target, dtype=np.int).mean()

        return accr

    def _accuracy_calcluator_smiles(self, prediction: str, target: str):
        pass

