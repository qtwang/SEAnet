# coding = utf-8
# modified from https://github.com/hfawaz/InceptionTime/blob/458e5caf2093762fe2ac38f701304a600a08e6d7/classifiers/inception.py

from typing import List
from collections import OrderedDict

import torch
from torch import nn, Tensor
import numpy as np

from util.conf import Configuration
from model.commons import Squeeze, Reshape



class _InceptionLayer(nn.Module):
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, dilation: int = 1):
        super(_InceptionLayer, self).__init__()

        padding = int(kernel_size / 2) * dilation

        # TODO bottleneck is more commonly inside layer
        # bottleneck_channels = conf.getHP('inception_bottleneck_channels')
        # self.__bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        # self.__convolution = nn.Conv1d(bottleneck_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        
        self.__convolution = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)


    def forward(self, input) -> Tensor:
        # bottleneck = self.__bottleneck(input)
        # return self.__convolution(bottleneck)

        return self.__convolution(input)


class _InceptionBlock(nn.ModuleDict):
    def __init__(self, conf: Configuration, in_channels: int, out_channels: int, to_encode: bool):
        super(_InceptionBlock, self).__init__()

        # TODO bottleneck is more commonly inside layer
        bottleneck_channels = conf.getHP('inception_bottleneck_channels')
        self.__bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)

        inception_kernel_sizes = np.sort(np.array(conf.getHP('inception_kernel_sizes')))
        num_layers = len(inception_kernel_sizes)

        layer_chennels = np.array([out_channels // num_layers] * num_layers)
        
        for i in range(out_channels % num_layers):
            layer_chennels[i] += 1

        assert np.sum(layer_chennels) == out_channels
        assert inception_kernel_sizes[0] == 1

        self.__identity_link = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                             nn.Conv1d(in_channels, layer_chennels[0], 1, bias=False))

        for i, (kernel_size, layer_out_channels) in enumerate(zip(inception_kernel_sizes[1: ], layer_chennels[1: ])):
            self.add_module('InceptionLayer{:d}'.format(i), _InceptionLayer(kernel_size, bottleneck_channels, layer_out_channels))

        self.__finalize = nn.Sequential(nn.BatchNorm1d(out_channels),
                                        conf.getActivation(conf.getHP('activation_conv')))


    def forward(self, input: Tensor) -> Tensor:
        latent = [self.__identity_link(input)]

        bottleneck = self.__bottleneck(input)

        for label, layer in self.items():
            if label.startswith('InceptionLayer'):
                latent.append(layer(bottleneck))
            
        latent = torch.cat(latent, 1)

        return self.__finalize(latent)



class _InceptionNet(nn.Module):
    def __init__(self, conf: Configuration, to_encode: bool):
        super(_InceptionNet, self).__init__()

        num_chennels = conf.getHP('num_en_channels') if to_encode else conf.getHP('num_de_channels')
        num_blocks = conf.getHP('num_inceptionen_blocks') if to_encode else conf.getHP('num_inceptionde_blocks')
        assert num_blocks >= 1

        self.__model = nn.Sequential(OrderedDict([
                           ('InceptionBlock0', _InceptionBlock(conf, 1, num_chennels, to_encode))
                       ]))

        for depth in range(1, num_blocks):
            block = _InceptionBlock(conf, num_chennels, num_chennels, to_encode)
            self.__model.add_module('InceptionBlock{:d}'.format(depth), block)


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class InceptionEncoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(InceptionEncoder, self).__init__()

        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_en_channels')
        dim_latent = conf.getHP('dim_en_latent')

        self.__model = nn.Sequential(_InceptionNet(conf, to_encode=True),
                                     nn.AdaptiveMaxPool1d(1),
                                     Squeeze(),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_embedding, bias=False),
                                     nn.LayerNorm(dim_embedding, elementwise_affine=False) if conf.getHP('encoder_normalize_embedding') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class InceptionDecoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(InceptionDecoder, self).__init__()

        dim_series = conf.getHP('dim_series')
        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_de_channels')
        dim_latent = conf.getHP('dim_de_latent')

        self.__model = nn.Sequential(Reshape([-1, 1, dim_embedding]),
                                     nn.Linear(dim_embedding, dim_series),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     _InceptionNet(conf, to_encode=False),
                                     nn.AdaptiveMaxPool1d(1),
                                     Reshape([-1, 1, num_channels]),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_series, bias=False),
                                     nn.LayerNorm(dim_series, elementwise_affine=False) if conf.getHP('decoder_normalize_reconstruction') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)
