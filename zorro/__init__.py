import math
import torch
import torch.nn as nn
from torch import Tensor
from models.common import Conv, Bottleneck


class Zorro(nn.Module):
    param_a = 1.2
    param_b = 1
    const_k = 1 + math.exp(param_a * param_b)
    const_0 = torch.zeros(1)
    const_1 = torch.ones(1)

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        # input = torch.tensor(input).to(device='cuda')
        x = torch.maximum(abs(input - 0.5) - 0.5, self.const_0)
        sigmoid = 1 / (1 + torch.exp(self.param_a * (x + self.param_b)))
        return torch.minimum(torch.maximum(input, self.const_0), self.const_1) + self.const_k * x * sigmoid * torch.sign(input)


class ConvZorro(Conv):
    default_act = Zorro()


class C3Zorro(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvZorro(c1, c_, 1, 1)
        self.cv2 = ConvZorro(c1, c_, 1, 1)
        self.cv3 = ConvZorro(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
