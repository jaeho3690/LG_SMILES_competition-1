import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from PIL import Image

from rdkit import Chem
from rdkit import DataStructs
from rdkit.DataStructs import FingerprintSimilarity as FPS
from rdkit.Chem import MolFromSmiles,RDKFingerprint

from model.Network import Encoder, DecoderWithAttention, PredictiveDecoder
from model.Predictor import Predict
from utils import make_directory, decode_predicted_sequences

import random
import numpy as np
import yaml
import asyncio
import os

from itertools import combinations
from collections import Counter
import warnings


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
        self._test_file_path = config.test_file_path

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
            self._decoder.to(self._device)
        self._encoder = Encoder()
        self._encoder.to(self._device)

        self._encoder.fine_tune(self._fine_tune_encoder)
        self._encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                 self._encoder.parameters()),
                                                   lr=self._encoder_lr) if self._fine_tune_encoder else None
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self._encoder = nn.DataParallel(self._encoder)
        self._criterion = nn.CrossEntropyLoss().to(self._device)

    def _clip_gradient(self, optimizer, grad_clip):

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

    def model_test(self, submission, data_list, reversed_token_map, transform):

        self._encoder.eval()
        self._decoder.eval()

        for i, dat in enumerate(data_list):
            imgs = Image.open(self._test_file_path + dat)
            imgs = self.png_to_tensor(imgs)
            imgs = transform(imgs).to(self._device)

            predictions = None
            decoded_sequences = None
            is_smiles = False
            add_seed = 0

            while not is_smiles:
                self._seed_everything(self._seed + add_seed)
                encoded_imgs = self._encoder(imgs.unsqueeze(0))
                predictions = self._decoder(encoded_imgs, self._decode_length)

                SMILES_predicted_sequence = list(torch.argmax(predictions.detach().cpu(), -1).numpy())[0]
                decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, reversed_token_map)

                is_smiles = self.is_smiles(decoded_sequences)
                add_seed += 1

                print('{} is {}, {} seed sequence:, {}'.format(i, is_smiles, add_seed, decoded_sequences))
                break

            submission.loc[submission['file_name']== dat, 'SMILES'] = decoded_sequences
            del (predictions)

        return submission


    def ensemble_test(self, submission, data_list, reversed_token_map, transform):
        predictors = []
        encoder_type = ['wide_res', 'wide_res', 'res', 'res']
        with open('model/prediction_models.yaml') as f:
            p_configs = yaml.load(f)

        for conf in p_configs.values():
            predictors.append(Predict(conf, reversed_token_map,
                                      self._decode_length, self._model_load_path))

        fault_counter = 0

        for i, dat in enumerate(data_list):
            imgs = Image.open(self._test_file_path + dat)
            imgs = self.png_to_tensor(imgs)
            imgs = transform(imgs).to(self._device)

            # predict SMILES sequence form eqch predictors
            preds = []
            for p in predictors:
                preds.append(p.SMILES_prediction(imgs))

            print('preds:', preds)

            # fault check
            ms = {}
            for idx, p in enumerate(preds):
                m = MolFromSmiles(p)
                if m != None:
                    ms.update({idx:m})

            if len(ms) == 0:
                fault_counter += 1
                ms.update({0:preds[0]})

            top_k = 3
            # result ensemble
            ms_to_fingerprint = [RDKFingerprint(x) for x in ms.values()]
            combination_of_smiles = list(combinations(ms_to_fingerprint, 2))
            ms_to_index = [x for x in ms]
            combination_index = list(combinations(ms_to_index, 2))

            print('combination_of_smiles:', combination_of_smiles)
            print('combination_index:', combination_index)

            smiles_dict = {}
            for combination, index in zip(combination_of_smiles, combination_index):
                smiles_dict[index] = (FPS(combination[0], combination[1]))

            # sort by score
            smiles_dict = sorted(smiles_dict.items(), key=(lambda x: x[1]), reverse=True)

            most_common_k = Counter(smiles_dict).most_common(top_k)

            print('smiles_dict:', smiles_dict)
            print('most_common_k:', most_common_k)

            if list(smiles_dict.values()) == 1.0:
                sequence = preds[combination_index[0][0][1]]
            else:
                first_common = Counter(most_common_k[0][0])
                second_common = Counter(most_common_k[1][0])
                combine = first_common + second_common
                sequence = preds[combine.most_common(1)[0][0]]

            print('{} sequence:, {}'.format(i, sequence))

            submission.loc[submission['file_name'] == dat, 'SMILES'] = sequence
            del(preds)

        return submission


    # def model_test_old(self, submission, test_loader, reversed_token_map):
    #
    #     self.model_load()
    #     self._encoder.eval()
    #     self._decoder.eval()
    #
    #     for i, imgs in enumerate(test_loader):
    #         imgs = imgs.to(self._device)
    #
    #         imgs = self._encoder(imgs)
    #         predictions = self._decoder(imgs)
    #         SMILES_predicted_sequence = list(torch.argmax(predictions.detach().cpu(), -1).numpy())[0]
    #         decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, reversed_token_map)
    #         print('{}:, {}'.format(i, decoded_sequences))
    #         submission['SMILES'].loc[i] = decoded_sequences
    #         del (predictions)
    #
    #     return submission

    def png_to_tensor(self, img: Image):
        img = img.resize((256,256))
        img = np.array(img)
        img = np.moveaxis(img, 2, 0)
        return torch.FloatTensor(img) / 255.

    def is_smiles(self, sequence):
        m = Chem.MolFromSmiles(sequence)
        return False if m == None else True

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

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
