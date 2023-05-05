import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # attn.shape=[batch_size, n_head, q_len, k_len]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn,dim=1))
        out = torch.matmul(attn, v)   # out.shape=[batch_size, n_head, q_len, d_v]
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass


