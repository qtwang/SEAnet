# coding = utf-8

import numbers
from typing import Union, List

import torch
from torch import Tensor, Size
from torch.nn import Module, init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.functional import normalize


# cite: https://github.com/lancopku/AdaNorms
# cite: NeurIPS19 Understanding and Improving Layer Normalization
class AdaNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', 'k', 'scale']
    
    normalized_shape: Union[int, List[int], torch.Size]
    eps: float
    elementwise_affine: bool
    k: float
    scale: float

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], k: float = 1 / 10, scale: float = 2., eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(AdaNorm, self).__init__()

        self.k = k
        self.scale = scale
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        else:
            raise ValueError('Only last layer for AdaNorm currently')
        self.normalized_shape = tuple(normalized_shape)

        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)


    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)

        input = input - mean
        mean = input.mean(-1, keepdim=True)
        
        graNorm = (self.k * (input - mean) / (std + self.eps)).detach()
        input_norm = (input - input * graNorm) / (std + self.eps)
        
        return self.scale * input_norm


    def extra_repr(self) -> Tensor:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, k={k}, scale={scale}'.format(**self.__dict__)



# regulize (preserve) orthogonality among output features/channels 
# under Spectral Restricted Isometry Property (of orthogonal matrix)
# extra hyper-parameters is added and should be searched: coefficient (weight) of SRIPTerm
# recommended by the authors: 1e-1(epoch 0) --> 1e-3(20) --> 1e-4(50) --> 1e-6(70) --> 0(120) of 200 epochs totally
# while at the same time changing coefficient of weight decay: 1e-8(0) --> 1e-4(20)
# cite: https://github.com/VITA-Group/Orthogonality-in-CNNs/blob/master/Imagenet/resnet/train_n.py
# cite: NeurIPS18 Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
def getSRIPTerm(model: Module, device='cpu'):
    term = None

    for W in model.parameters():
        if W.ndimension() < 2:
            continue
        else:
            # for convolutional:
            # W.shape = [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE] 
            # rows = OUTPUT_CHANNELS, cols = INPUT_CHANNELS * KERNEL_SIZE

            # for linner:
            # W.shape = [OUTPUT_FEATURES, INTPUT_FEATURES]
            # rows = OUTPUT_FEATURES, cols = INTPUT_FEATURES

            cols = W[0].numel()
            rows = W.shape[0]

            w1 = W.view(-1, cols)
            wt = torch.transpose(w1, 0, 1)
            m  = torch.matmul(wt, w1)

            ident = Variable(torch.eye(cols,cols))
            ident = ident.to(device)

            w_tmp = (m - ident)
            height = w_tmp.size(0)

            # iterative computing approximate sigma
            u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)

            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if term is None:
                term = (sigma) ** 2
            else:
                term = term + (sigma) ** 2
                
    return term

