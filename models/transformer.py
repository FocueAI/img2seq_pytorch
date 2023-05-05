# https://github.com/jadore801120/attention-is-all-you-need-pytorch 参考该作者的源码
import torch
from torch import nn


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class Transformer(nn.Module):
    def __init__(self, d_model, n_trg_vocab):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

    def forward(self, src_seq, trg_seq):
        encoder_out = self.encoder(src_seq)
        decoder_out = self.decoder(encoder_out, trg_seq)
        seq = self.trg_word_prj(decoder_out)

        return seq

if __name__ == '__main__':
    src_seq = ['boc','a' ,'b' ,'eoc']
    Transformer()