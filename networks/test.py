import torch
from torch import nn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


x = [[1, 2, 3, 4],
     [3, 5, 2, 1]]
x = torch.Tensor(x)
print(x)
y = maxpool(x, dim=1)
print(y)
