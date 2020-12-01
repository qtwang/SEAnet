# coding = utf-8

from torch import nn, Tensor


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input: Tensor) -> Tensor:
        return input.view(self.shape)



class Squeeze(nn.Module):
    def __init__(self, dim = None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if self.dim is None:
            return input.squeeze()
        else:
            return input.squeeze(dim=self.dim)

