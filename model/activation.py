# coding = utf-8

import numpy as np
from torch import nn, tanh, Tensor
    
EPSILON = 1e-7


def EQUAL(a:float, b:float):
    return (a - b <= EPSILON) and (b - a <= EPSILON)


class TanhAdjusted(nn.Module):
    def __init__(self, outer:float=1., inner:float=1., specified=False):
        super(TanhAdjusted, self).__init__()

        self.a = outer
        self.b = inner

        if not specified:
            if EQUAL(self.a, 1.) and not EQUAL(self.b, 1.):
                self.a = 1. / np.tanh(self.b)
            elif not EQUAL(self.a, 1.) and EQUAL(self.b, 1.):
                self.b = np.log((self.a + 1.) / (self.a - 1.)) / 2.
            

    def forward(self, input: Tensor) -> Tensor:
        return self.a * tanh(self.b * input)



class LeCunTanh(nn.Module):
    def __init__(self):
        super(LeCunTanh, self).__init__()

        self.adjustedTanh = TanhAdjusted(outer=1.7159, inner=2/3, specified=True)
            

    def forward(self, input: Tensor) -> Tensor:
        return self.adjustedTanh(input)



if __name__ == "__main__":
    print('Welcome to where the activations got defined!')
