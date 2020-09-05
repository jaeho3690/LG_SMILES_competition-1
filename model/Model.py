import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from model.Network import Encoder, DecoderWithAttention

class MSTS:
    def __init__(self, config):
        # self._data_folder = config.data_folder
        # self._data_name = config.data_name

        self._vocab_size = 50
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
        # self._checkpoint = config.checkpoint

        self._decoder = DecoderWithAttention(attention_dim=self._attention_dim,
                                             embed_dim=self._emb_dim,
                                             decoder_dim=self._decoder_dim,
                                             vocab_size=self._vocab_size,
                                             dropout=self._dropout)
        self._decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                           self._decoder.parameters()),
                                                           lr=self._decoder_lr)
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
            self._decoder = nn.DataParallel(self._decoder)
        self._criterion = nn.CrossEntropyLoss().to(self._device)

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

        loss = None

        for i, (imgs, caps, caplens) in enumerate(train_loader):
            print(i,end='\r')
            imgs = imgs.to(self._device)
            caps = caps.to(self._device)
            caplens = caplens.to(self._device)

            # Forward prop.
            imgs = self._encoder(imgs)
            predictions, caps_sorted, decode_lengths, alphas, sort_ind = self._decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            total_length = decode_lengths.size(1)
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True, total_length=total_length).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, total_length=total_length).data

            # Calculate loss
            loss = self._criterion(predictions, targets)

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

        return loss

    def model_save(self, save_num):
        torch.save(
            self._decoder.state_dict(),
            'graph_save/decoder{}.pkl'.format(str(save_num).zfill(3))
        )
        torch.save(
            self._encoder.state_dict(),
            'graph_save/encoder{}.pkl'.format(str(save_num).zfill(3))
        )


    def mode_load(self, load_num):
        self._decoder.load_state_dict(
            torch.load('graph_save/decoder{}.pkl'.format(str(load_num).zfill(3)))
        )
        self._encoder.load_state_dict(
            torch.load('graph_save/encoder{}.pkl'.format(str(load_num).zfill(3)))
        )
