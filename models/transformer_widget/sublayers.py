import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ softmax(Q*K)*V, 从而实现对V产生不同大小的权重 """

    def __init__(self, temperature, attn_dropout=0.1):
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # TODO: 会后要评价出2者的区别, 看是否能结合出2者的优势
        # 写法一:
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # attn.shape=[batch_size, n_head, q_len, k_len]
        # 写法二:
        # attn = torch.matmul(q * k.transpose(2,3))/np.sqrt(k.size(-1))   # k.size(-1) 表示 d_k

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=1))
        out = torch.matmul(attn, v)  # out.shape=[batch_size, n_head, q_len, d_v]
        return out


class MultiHeadAttention(nn.Module):
    """ 感觉这部分代码是核心 """
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        """
        :param d_model: transformer的模型维度
        :param n_head: 注意力的头数
        :param d_k:    Q与K 的 维度
        :param d_v:    V的维度
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_head * d_k)
        self.W_K = nn.Linear(d_model, n_head * d_k)
        self.W_V = nn.Linear(d_model, n_head * d_v)
        self.linear = nn.Linear(n_head * d_v,
                                d_model)  # 一段操作后,要把输入维度还原成[batch, seq_len, d_model],  符合编码器与解码器输入输出维度不变的逻辑
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # 有些版本中有dropout, 有些版本中没有
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        :param q: [batch_size, len_q, d_model]
        :param k: [batch_size, len_k, d_model]
        :param v: [batch_size, len_k, d_model]
        :param mask: [batch_size, len_q, len_k]
        :return:
        """
        batch_size = q.size(0)
        len_q = q.size(1)
        len_k = v.size(2)

        q_s = self.W_K(k).view(batch_size, self.n_head, len_q, self.d_k)
        k_s = self.W_K(k).view(batch_size, self.n_head, len_k, self.d_k)
        v_s = self.W_K(k).view(batch_size, self.n_head, len_k, self.d_v)

        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        attn = self.attention(q_s, k_s, v_s, mask=mask)  # shape: [batch_size, n_head, q_len, d_v]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, len_q, self.n_head * self.d_v)
        out = self.dropout(self.linear(attn))  # [batch_size, len_q, d_model]
        return out
