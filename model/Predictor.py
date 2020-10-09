import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model.Network import Encoder, PredictiveDecoder
from utils import decode_predicted_sequences

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Predict():
    def __init__(self, config, reversed_token_map, decode_length, load_path):

        self._vocab_size = 70
        self._decode_length = decode_length
        self._emb_dim = int(config['emb_dim'])
        self._attention_dim = int(config['attention_dim'])
        self._decoder_dim = int(config['decoder_dim'])
        self._reversed_token_map = reversed_token_map

        self._model_load_name = config['load_model_name']
        self._model_load_path = load_path

        self._encoder = Encoder(model_type=config['encoder_type'])
        self._decoder = PredictiveDecoder(attention_dim=self._attention_dim,
                                          embed_dim=self._emb_dim,
                                          decoder_dim=self._decoder_dim,
                                          vocab_size=self._vocab_size)

        self._encoder.to(device)
        self._decoder.to(device)

        self.model_load()
        print(self._model_load_name, 'load successed!')

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

        weight_data = torch.load('{}/encoder{}.pkl'.format(self._model_load_path, self._model_load_name))
        new_keys = [x[7:] for x in list(weight_data.keys())]
        encoder_weight = {}
        for key, n_key in zip(weight_data.keys(), new_keys):
            encoder_weight[n_key] = weight_data[key]
        self._encoder.load_state_dict(encoder_weight)
