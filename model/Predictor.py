import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model.Network import Encoder, PredictiveDecoder
from utils import decode_predicted_sequences


class Predict():
    def __init__(self, config, load_model_name, reversed_token_map, decode_length, load_path):

        self._vocab_size = 70
        self._decode_length = decode_length
        self._emb_dim = int(config['emb_dim'])
        self._attention_dim = int(config['attention_dim'])
        self._decoder_dim = int(config['decoder_dim'])
        self._reversed_token_map = reversed_token_map

        self._model_load_name = load_model_name
        self._model_load_path = load_path

        self._encoder = Encoder(config['encoder_type'])
        self._decoder = PredictiveDecoder(attention_dim=self._attention_dim,
                                          embed_dim=self._emb_dim,
                                          decoder_dim=self._decoder_dim,
                                          vocab_size=self._vocab_size)
        self.model_load()
        print(load_model_name, 'load successed!')

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
            torch.load('{}/decoder{}.pkl'.format(self._model_load_path, self._model_load_name))
        )
        self._encoder.load_state_dict(
            torch.load('{}/encoder{}.pkl'.format(self._model_load_path, self._model_load_name))
        )