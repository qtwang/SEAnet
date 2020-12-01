# cite: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/4baa655c3a761001f100c4a9955e57dc93589957/networks/causal_cnn.py
# which cites: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# coding = utf-8

from torch import nn, Tensor

from util.conf import Configuration
from model.commons import Squeeze, Reshape


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, input: Tensor) -> Tensor:
        return input[:, :, : -self.chomp_size]


class _OriginalResBlock(nn.Module):
    def __init__(self, conf: Configuration, in_channels, out_channels, dilation, final: bool = False):
        super(_OriginalResBlock, self).__init__()

        kernel_size = conf.getHP('size_kernel')
        padding = (kernel_size - 1) * dilation
        activation_name = conf.getHP('activation_conv')
        bias = conf.getHP('layernorm_type') == 'none'

        self.__residual_link = nn.Sequential(conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                             Chomp1d(padding),
                                             conf.getActivation(activation_name),

                                             conf.getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                             Chomp1d(padding),
                                             conf.getActivation(activation_name))
        
        if in_channels != out_channels:
            self.__identity_link = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        else:
            self.__identity_link = nn.Identity()

        self.__after_addition = conf.getActivation(activation_name) if final else nn.Identity()
        
        
    def forward(self, input: Tensor) -> Tensor:
        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)



# class _PreActivatedResBlock(nn.Module):
#     def __init__(self, conf: Configuration, in_channels, out_channels, dilation, first = False, last = False):
#         super(_PreActivatedResBlock, self).__init__()

#         dim_series = conf.getHP('dim_series')
#         kernel_size = conf.getHP('size_kernel')
#         padding = int(kernel_size / 2) * dilation
#         activation_name = conf.getHP('activation_conv')
#         bias = conf.getHP('layernorm_type') == 'none' or not conf.getHP('layernorm_elementwise_affine')

#         if first:
#             self.__first_block = conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias))
#             in_channels = out_channels
#         else:
#             self.__first_block = nn.Identity()

#         self.__residual_link = nn.Sequential(conf.getLayerNorm(dim_series), 
#                                              conf.getActivation(activation_name),
#                                              conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                      
#                                              conf.getLayerNorm(dim_series),
#                                              conf.getActivation(activation_name),
#                                              conf.getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)))
        
#         if in_channels != out_channels:
#             self.__identity_link = conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, 1, bias=bias))
#         else:
#             self.__identity_link = nn.Identity()

#         if last:
#             self.__after_addition = nn.Sequential(conf.getLayerNorm(dim_series), 
#                                                    conf.getActivation(activation_name))
#         else:
#             self.__after_addition = nn.Identity()
        
        
#     def forward(self, input: Tensor) -> Tensor:
#         input = self.__first_block(input)

#         residual = self.__residual_link(input)
#         identity = self.__identity_link(input)

#         return self.__after_addition(identity + residual)



class _ResNet(nn.Module):
    def __init__(self, conf: Configuration, to_encode: bool):
        super(_ResNet, self).__init__()

        num_resblock = conf.getHP('num_en_resblock') if to_encode else conf.getHP('num_de_resblock')

        if conf.getHP('dilation_type') == 'exponential':
            assert num_resblock > 1 and 2 ** (num_resblock + 1) <= conf.getHP('dim_series') + 1

        inner_channels = conf.getHP('num_en_channels') if to_encode else conf.getHP('num_de_channels')
        out_channels = conf.getHP('dim_en_latent') if to_encode else conf.getHP('dim_de_latent')

        if conf.getHP('resblock_pre_activation'):
            raise ValueError('pre-activated FDJ is not yet supported')
            # layers = [_PreActivatedResBlock(conf, 1, inner_channels, conf.getDilatoin(1, to_encode), first=True)]
            # layers += [_PreActivatedResBlock(conf, inner_channels, inner_channels, conf.getDilatoin(depth, to_encode)) for depth in range(2, num_resblock)]
            # layers += [_PreActivatedResBlock(conf, inner_channels, out_channels, conf.getDilatoin(num_resblock, to_encode), last=True)]
        else:
            layers = [_OriginalResBlock(conf, 1, inner_channels, conf.getDilatoin(1, to_encode))]
            layers += [_OriginalResBlock(conf, inner_channels, inner_channels, conf.getDilatoin(depth, to_encode)) for depth in range(2, num_resblock)]
            layers += [_OriginalResBlock(conf, inner_channels, out_channels, conf.getDilatoin(num_resblock, to_encode), final=True)]

        self.__model = nn.Sequential(*layers)

        
    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class FDJEncoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(FDJEncoder, self).__init__()

        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_en_channels')
        dim_latent = conf.getHP('dim_en_latent')

        self.__model = nn.Sequential(_ResNet(conf, to_encode=True),
                                     nn.AdaptiveMaxPool1d(1),
                                     Squeeze(),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_embedding, bias=False),
                                     nn.LayerNorm(dim_embedding, elementwise_affine=False) if conf.getHP('encoder_normalize_embedding') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class FDJDecoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(FDJDecoder, self).__init__()

        dim_series = conf.getHP('dim_series')
        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_de_channels')
        dim_latent = conf.getHP('dim_de_latent')

        self.__model = nn.Sequential(Reshape([-1, 1, dim_embedding]),
                                     nn.Linear(dim_embedding, dim_series),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     _ResNet(conf, to_encode=False),
                                     nn.AdaptiveMaxPool1d(1),
                                     Reshape([-1, 1, num_channels]),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_series, bias=False),
                                     nn.LayerNorm(dim_series, elementwise_affine=False) if conf.getHP('decoder_normalize_reconstruction') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)
