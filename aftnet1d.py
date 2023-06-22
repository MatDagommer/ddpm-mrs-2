import torch
from torch import nn, Tensor
from torch.nn import Module

from cplxtorch import nn as cnn
from cplxtorchvision.models import UNet1d


def basicblock(in_features: int, inter_features: int, out_features: int) -> Module:
    return nn.Sequential(
        cnn.Linear(in_features, inter_features),
        cnn.Linear(inter_features, out_features)
    )


class AFT(Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.linear = basicblock(features, features, features)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input)


class RACUNet(Module):
    """ Residual Attention Complex-valued UNet """
    def __init__(self) -> None:
        super().__init__()
        self.cunet = UNet1d(1, 1, [32, 64, 128, 256, 512], True) # in_channels, out_channels, channels for each layer, attention

    def forward(self, input: Tensor) -> Tensor:
        t = self.cunet(input)
        output = input + t
        return output


class AFT_RACUNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.aft = AFT(2048) # input signal length
        self.racunet = RACUNet()

    def forward(self, input: Tensor) -> Tensor:
        t = self.aft(input)
        output = self.racunet(t)
        return output