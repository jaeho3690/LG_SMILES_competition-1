import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
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


class Predict():
    def __init__(self, config, load_model_name, encoder_type, reversed_token_map):

        self._vocab_size = 70
        self._decode_length = config.decode_length
        self._emb_dim = config.emb_dim
        self._attention_dim = config.attention_dim
        self._decoder_dim = config.decoder_dim
        self._reversed_token_map = reversed_token_map

        self._model_load_name = load_model_name

        self._encoder = Encoder(encoder_type)
        self._decoder = PredictiveDecoder(attention_dim=self._attention_dim,
                                          embed_dim=self._emb_dim,
                                          decoder_dim=self._decoder_dim,
                                          vocab_size=self._vocab_size)
        self.model_load()

    def SMILES_prediction(self, img):

        self._encoder.eval()
        self._decoder.eval()

        encoded_img = self._encoder(img.unsqueeze(0))
        predictions = self._decoder(encoded_img, self._decode_length)

        SMILES_predicted_sequence = list(torch.argmax(predictions.detach().cpu(), -1).numpy())[0]
        decoded_sequences = decode_predicted_sequences(SMILES_predicted_sequence, self._reversed_token_map)

        return decoded_sequences


    def model_load(self):
        self._decoder.load_state_dict(
            torch.load('{}/decoder{}.pkl'.format(self._model_load_name))
        )
        self._encoder.load_state_dict(
            torch.load('{}/encoder{}.pkl'.format(self._model_load_name))
        )