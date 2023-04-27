from typing import Set

import editdistance
import torch
from torch import Tensor
from torchmetrics import Metric
from torch import nn

# 新修改的
class CharacterErrorRate(nn.Module):
    def __init__(self, ignore_indices: Set[int], *args):
        super().__init__(*args)
        self.ignore_indices = ignore_indices
        self.error = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)


    def forward(self, preds, targets):
        N = preds.shape[0]
        for i in range(N):
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]
            distance = editdistance.distance(pred, target)
            if max(len(pred), len(target)) > 0:
                self.error += distance / max(len(pred), len(target))
        self.total += N

    def compute(self) -> Tensor:
        return self.error / self.total

    def reset(self):
        self.error -= self.error
        self.total -= self.total

# 原始的
# class CharacterErrorRate(Metric):
#     def __init__(self, ignore_indices: Set[int], *args):
#         super().__init__(*args)
#         self.ignore_indices = ignore_indices
#         self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.error: Tensor
#         self.total: Tensor
#
#     def update(self, preds, targets):
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
