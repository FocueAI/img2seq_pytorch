import torch
from torch import nn
from torch import Tensor
from typing import List,Set
import editdistance
from torchmetrics import Metric
import torchvision.models
from utils.data_deal.dataset import Tokenizer
from models.resnet_transformer import ResNetTransformer
from utils.metrics_zoo.char_metric import CharacterErrorRate


# class Img2Seq(nn.Module):
def Img2Seq(d_model: int,
            dim_feedforward: int,
            nhead: int,
            dropout: float,
            num_decoder_layers: int,
            max_output_len: int,
            vocab_path: str
            ):
    tokenizer = Tokenizer.load(vocab_path)
    # ------------------------------构建模型-------------------------------- #
    # encode(resNet18) + decode(transformer_decoder)
    model = ResNetTransformer(
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        nhead=nhead,
        dropout=dropout,
        num_decoder_layers=num_decoder_layers,
        max_output_len=max_output_len,
        sos_index=tokenizer.sos_index,
        eos_index=tokenizer.eos_index,
        pad_index=tokenizer.pad_index,
        num_classes=len(tokenizer)
    )
    # -----------------------------损失函数------------------------------------ #
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    # val_cer = CharacterErrorRate()
    # test_cer = CharacterErrorRate(tokenizer.ignore_indices)
    return model, loss_fn, tokenizer

# class CharacterErrorRate(nn.Module):
#     def __init__(self, ignore_indices: Set[int]=tokenizer.pad_index, *args):
#         super().__init__(*args)
#         self.ignore_indices = ignore_indices
#         self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.error: Tensor
#         self.total: Tensor
#
#     def forward(self, preds, targets):
#         N = preds.shape[0]
#         for i in range(N):
#             pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
#             target = [token for token in targets[i].tolist() if token not in self.ignore_indices]
#             distance = editdistance.distance(pred, target)
#             if max(len(pred), len(target)) > 0:
#                 self.error += distance / max(len(pred), len(target))
#         self.total += N
#
#     def compute(self) -> Tensor:
#         return self.error / self.total
#
#     def reset(self):
#         self.error -= self.error
#         self.total -= self.total


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

        self.correct = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):  # 在后续中,每一步的损失都是调用这里的
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0]
        self.correct += m
        self.total += n

        return m / n

    def compute(self):
        return self.correct.float() / self.total

    def reset(self):
        self.correct -= self.correct
        self.total -= self.total


if __name__ == '__main__':
    net = Img2Seq()
    input = torch.zeros((8, 1, 32, 32))
    out = net(input)
    print(f'out-shape:{out.shape}')
