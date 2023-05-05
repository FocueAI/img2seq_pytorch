import math
from re import I
from typing import Union

import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor

from .positional_encoding import PositionalEncoding1D, PositionalEncoding2D
from .resnet import ResNet, ResBlock






class ResNetTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        max_output_len: int,
        sos_index: int,
        eos_index: int,
        pad_index: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_output_len = max_output_len + 2
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index

        # Encoder
        # resnet = torchvision.models.resnet18(pretrained=False)  # 直接使用 torch现有的集成的模型结构
        # self.backbone = nn.Sequential(                          # 和论文中的一致,只使用resnet18的前4个卷积层
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool,
        #     resnet.layer1,
        #     resnet.layer2,
        #     nn.Conv2d(128, 256, 1) 
        #     # resnet.layer3,
        # )
        # self.bottleneck = nn.Conv2d(256, self.d_model, 1)       # 和论文一致使用了1*1卷积
        # self.bottleneck = nn.Conv2d(128, self.d_model, 1)       # 和论文一致使用了1*1卷积
        self.backbone = ResNet(ResBlock)
        self.bottleneck = nn.Conv2d(512, self.d_model, 1) 
        
        
        
        self.image_positional_encoder = PositionalEncoding2D(self.d_model)                                      # 编码器输入对应的2d位置编码

        # Decoder
        self.embedding = nn.Embedding(num_classes, self.d_model)                                                # num_classes = 字典的大小 ???????
        self.y_mask = generate_square_subsequent_mask(self.max_output_len)                                      # 防止学习到未来的词汇
        self.word_positional_encoder = PositionalEncoding1D(self.d_model, max_len=self.max_output_len)          # 解码器输入对应的1d位置编码
        transformer_decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_decoder_layers)
        self.fc = nn.Linear(self.d_model, num_classes)

        # It is empirically important to initialize weights properly
        if self.training:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

        nn.init.kaiming_normal_(
            self.bottleneck.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.bottleneck.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.bottleneck.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.bottleneck.bias, -bound, bound)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, _E, _H, _W)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (B, num_classes, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, num_classes)
        output = output.permute(1, 2, 0)  # (B, num_classes, Sy)
        return output

    def encode(self, x: Tensor) -> Tensor:
        """Encode inputs.

        Args:
            x: (B, C, _H, _W)

        Returns:
            (Sx, B, E)
        """
        # Resnet expects 3 channels but training images are in gray scale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # 使用torch集成的resnet, 必须满足其使用条件, 输入数据的通道数为3的条件
        x = self.backbone(x)  # (B, RESNET_DIM, H, W); H = _H // 32, W = _W // 32
        x = self.bottleneck(x)  # (B, E, H, W)
        x = self.image_positional_encoder(x)  # (B, E, H, W)
        x = x.flatten(start_dim=2)  # (B, E, H * W)
        x = x.permute(2, 0, 1)  # (Sx, B, E); Sx = H * W
        return x

    def decode(self, y: Tensor, encoded_x: Tensor) -> Tensor:
        """Decode encoded inputs with teacher-forcing.

        Args:
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (Sy, B, num_classes) logits
        """
        # TODO:要给padding后的词加mask
        
        """
        def get_pad_mask(seq, pad_idx):
            return (seq != pad_idx).unsqueeze(-2)
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.d_model)  # (Sy, B, E)
        y = self.word_positional_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, num_classes)
        return output

    def predict(self, x: Tensor) -> Tensor:
        """Make predctions at inference time.

        Args:
            x: (B, C, H, W). Input images.

        Returns:
            (B, max_output_len) with elements in (0, num_classes - 1).
        """
        B = x.shape[0]
        S = self.max_output_len

        encoded_x = self.encode(x)  # (Sx, B, E)

        output_indices = torch.full((B, S), self.pad_index).type_as(x).long()
        output_indices[:, 0] = self.sos_index
        has_ended = torch.full((B,), False)

        for Sy in range(1, S):
            y = output_indices[:, :Sy]  # (B, Sy)
            logits = self.decode(y, encoded_x)  # (Sy, B, num_classes)
            # Select the token with the highest conditional probability
            output = torch.argmax(logits, dim=-1)  # (Sy, B)
            output_indices[:, Sy] = output[-1:]  # Set the last output token

            # Early stopping of prediction loop to speed up prediction
            has_ended |= (output_indices[:, Sy] == self.eos_index).type_as(has_ended)
            if torch.all(has_ended):
                break

        # Set all tokens after end token to be padding
        eos_positions = find_first(output_indices, self.eos_index)
        for i in range(B):
            j = int(eos_positions[i].item()) + 1
            output_indices[i, j:] = self.pad_index

        return output_indices


def generate_square_subsequent_mask(size: int) -> Tensor:
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def find_first(x: Tensor, element: Union[int, float], dim: int = 1) -> Tensor:
    """Find the first occurence of element in x along a given dimension.

    Args:
        x: The input tensor to be searched.
        element: The number to look for.
        dim: The dimension to reduce.

    Returns:
        Indices of the first occurence of the element in x. If not found, return the
        length of x along dim.

    Usage:
        >>> first_element(Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])

    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/9

        I fixed an edge case where the element we are looking for is at index 0. The
        original algorithm will return the length of x instead of 0.
    """
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    indices[(~found) & (indices == 0)] = x.shape[dim]
    return indices
