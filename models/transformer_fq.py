# https://github.com/jadore801120/attention-is-all-you-need-pytorch 参考该作者的源码
import torch
from torch import nn
import numpy as np


# 其中的一种写法
def get_sinusoid_encoding_table(n_position, d_model):
    """ 原始论文中的1d的位置编码 """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


def get_attn_pad_mask(seq_q, seq_k):
    """

    :param seq_q: shape: [batch_size, len_q]
    :param seq_k: shape: [batch_size, len_k]
    :return: [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_q = seq_k.shape  # torch.shape 与 torch.size() 返回相同的值
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # shape=[batch_size, 1, len_k] # k是被询问的对象, 它里里面的padding部分不应当被注意,这个可以理解
                                                  # 但是 seq_k中的pad为什么不也屏蔽以下呢




class Encoder(nn.Module):
    """ transformer对应的编码器 """

    def __init__(self, src_vocab_size, d_model, max_src_len, n_layers, dropout=0.1):
        """
        :param src_vocab_size: 输入数据词汇表的大小
        :param d_model: transformer模型的维度
        :param max_src_len: transformer的encoder输入的最大序列的长度
        :param n_layers: encoder的层数
        """
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, n_position=max_src_len + 1)  # 根据位置编码公式,位置编码与pos和d_model有关
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer() for i in range(n_layers)])

    def forward(self, enc_inputs, scale_emb=False):
        """
        :param enc_inputs: [batch_size, seq_len] 每一个样本中的字符都是一个索引号  ----embedding----> [batch_size, seq_len, d_model]
        :param scale_emb: True:对 embedding 后的数据做缩小处理, False: 对 embedding 后的数组不做缩小处理
        :return: [batch_size, seq_len, d_model]
        """
        get
        enc_outputs = self.src_emb(enc_inputs)  # shape: [batch_size, seq_len, d_model]


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
    src_seq = ['boc', 'a', 'b', 'eoc']
    Transformer()
