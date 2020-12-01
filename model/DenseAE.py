# coding = utf-8
# modified from https://github.com/pytorch/vision/blob/c558be6b3b6ed5270ed2db0c5edc872c0d089c52/torchvision/models/densenet.py

from typing import List
from collections import OrderedDict

import torch
from torch import nn, Tensor

from util.conf import Configuration
from model.commons import Squeeze, Reshape



class _DenseLayer(nn.Module):
    def __init__(self, conf: Configuration, in_channels: int, dilation: int):
        super(_DenseLayer, self).__init__()

        dim_series = conf.getHP('dim_series')
        kernel_size = conf.getHP('size_kernel')
        padding = int(kernel_size / 2) * dilation
        activation_name = conf.getHP('activation_conv')
        bias = conf.getHP('layernorm_type') == 'none' or not conf.getHP('layernorm_elementwise_affine')

        growth_rate = conf.getHP('dense_growth_rate')
        bottleneck_multiplier = conf.getHP('dense_bottleneck_multiplier')
        bottleneck_channels = int(growth_rate * bottleneck_multiplier)

        self.__bottleneck = nn.Sequential(conf.getLayerNorm(dim_series), 
                                          conf.getActivation(activation_name),
                                          conf.getWeightNorm(nn.Conv1d(in_channels, bottleneck_channels, 1, bias=bias)))

        self.__convolution = nn.Sequential(conf.getLayerNorm(dim_series),
                                    conf.getActivation(activation_name),
                                    conf.getWeightNorm(nn.Conv1d(bottleneck_channels, growth_rate, kernel_size, padding=padding, dilation=dilation, bias=bias)))


    def forward(self, input) -> Tensor:
        if isinstance(input, List):
            input = torch.cat(input, 1)

        bottleneck = self.__bottleneck(input)

        return self.__convolution(bottleneck)



class _DenseBlock(nn.ModuleDict):
    def __init__(self, conf: Configuration, in_channels: int, dilation: int, num_layers: int):
        super(_DenseBlock, self).__init__()

        growth_rate = conf.getHP('dense_growth_rate')

        for i in range(num_layers):
            self.add_module('denselayer{:d}'.format(i + 1), _DenseLayer(conf, in_channels + i * growth_rate, dilation))


    def forward(self, input: Tensor) -> Tensor:
        latent = [input]

        for _, layer in self.items():
            latent.append(layer(latent))
            
        return torch.cat(latent, 1)



class _Transition(nn.Sequential):
    def __init__(self, conf: Configuration, in_channels: int):
        super(_Transition, self).__init__()

        bias = conf.getHP('layernorm_type') == 'none' or not conf.getHP('layernorm_elementwise_affine')

        self.add_module('normalize', conf.getLayerNorm(conf.getHP('dim_series')))
        self.add_module('activate', conf.getActivation(conf.getHP('activation_conv')))
        self.add_module('convolute', conf.getWeightNorm(nn.Conv1d(in_channels, conf.getHP('dense_transition_channels'), 1, bias=bias)))



class _DenseNet(nn.Module):
    def __init__(self, conf: Configuration, to_encode: bool):
        super(_DenseNet, self).__init__()

        kernel_size = conf.getHP('size_kernel')
        bias = conf.getHP('layernorm_type') == 'none' or not conf.getHP('layernorm_elementwise_affine')

        num_init_channels = conf.getHP('dense_init_channels')

        # DenseNet is by default pre-activation
        self.__model = nn.Sequential(OrderedDict([
                           ('conv0', conf.getWeightNorm(nn.Conv1d(1, num_init_channels, kernel_size, padding=int(kernel_size / 2), dilation=1, bias=bias)))
                       ]))

        num_blocks = conf.getHP('num_en_denseblocks') if to_encode else conf.getHP('num_de_denseblocks')
        num_layers = conf.getHP('num_en_denselayers') if to_encode else conf.getHP('num_de_denselayers')

        growth_rate = conf.getHP('dense_growth_rate')
        num_transition_channels = conf.getHP('dense_transition_channels')

        if conf.getHP('dilation_type') == 'exponential':
            assert num_blocks > 1 and 2 ** (num_blocks + 1) <= conf.getHP('dim_series') + 1

        in_channels = num_init_channels
        for depth in range(1, num_blocks + 1):
            denseblock = _DenseBlock(conf, in_channels, conf.getDilatoin(depth, to_encode), num_layers)
            self.__model.add_module('denseblock{:d}'.format(depth), denseblock)
            in_channels = in_channels + num_layers * growth_rate

            # different from original DenseNet, output_chennels is controlled as in the corresponding ResNet
            transblock = _Transition(conf, in_channels)
            self.__model.add_module('transition{:d}'.format(depth), transblock)
            in_channels = num_transition_channels

        # finalization of pre-activation mode
        self.__model.add_module('normalize', conf.getLayerNorm(conf.getHP('dim_series')))
        self.__model.add_module('activate', conf.getActivation(conf.getHP('activation_conv')))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class DenseEncoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(DenseEncoder, self).__init__()

        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_en_channels')
        dim_latent = conf.getHP('dim_en_latent')

        self.__model = nn.Sequential(_DenseNet(conf, to_encode=True),
                                     nn.AdaptiveMaxPool1d(1),
                                     Squeeze(),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_embedding, bias=False),
                                     nn.LayerNorm(dim_embedding, elementwise_affine=False) if conf.getHP('encoder_normalize_embedding') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class DenseDecoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(DenseDecoder, self).__init__()

        dim_series = conf.getHP('dim_series')
        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_de_channels')
        dim_latent = conf.getHP('dim_de_latent')

        self.__model = nn.Sequential(Reshape([-1, 1, dim_embedding]),
                                     nn.Linear(dim_embedding, dim_series),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     _DenseNet(conf, to_encode=False),
                                     nn.AdaptiveMaxPool1d(1),
                                     Reshape([-1, 1, num_channels]),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_series, bias=False),
                                     nn.LayerNorm(dim_series, elementwise_affine=False) if conf.getHP('decoder_normalize_reconstruction') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)
