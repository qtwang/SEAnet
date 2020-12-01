# coding = utf-8

from torch import nn, Tensor

from util.conf import Configuration
from model.ResidualAE import ResidualEncoder, ResidualDecoder, SingleResidualDecoder
from model.DenseAE import DenseEncoder, DenseDecoder
from model.RNNAE import RNNEncoder, RNNDecoder
from model.FDJAE import FDJEncoder, FDJDecoder
from model.InceptionAE import InceptionEncoder, InceptionDecoder


class AEBuilder(nn.Module):
    def __init__(self, conf: Configuration):
        super(AEBuilder, self).__init__()

        encoder_name = conf.getHP('encoder')

        if encoder_name == 'residual':
            self.__encoder = ResidualEncoder(conf)
        elif encoder_name == 'dense':
            self.__encoder = DenseEncoder(conf)
        elif encoder_name == 'fdj':
            self.__encoder = FDJEncoder(conf)
        elif encoder_name == 'inception':
            self.__encoder = InceptionEncoder(conf)
        elif encoder_name == 'gru' or encoder_name == 'lstm':
            self.__encoder = RNNEncoder(conf)
        else:
            raise ValueError('encoder {:s} isn\'t supported yet'.format(encoder_name))

        decoder_name = conf.getHP('decoder')

        if decoder_name == 'residual':
            self.__decoder = ResidualDecoder(conf)
        elif decoder_name == 'singleresidual':
            self.__decoder = SingleResidualDecoder(conf)
        elif decoder_name == 'dense':
            self.__decoder = DenseDecoder(conf)
        elif decoder_name == 'fdj':
            self.__decoder = FDJDecoder(conf)
        elif decoder_name == 'inception':
            self.__decoder = InceptionDecoder(conf)
        elif decoder_name == 'gru' or decoder_name == 'lstm':
            self.__decoder = RNNDecoder(conf)
        elif decoder_name == 'none':
            self.__decoder =  None
        else:
            raise ValueError('decoder {:s} isn\'t supported yet'.format(decoder_name))


    def encode(self, input: Tensor) -> Tensor:
        return self.__encoder(input)
    

    def decode(self, input: Tensor) -> Tensor:
        if self.__decoder is None:
            raise ValueError('No decoder')

        return self.__decoder(input)
    

    # explicit model.encode/decode is preferred as decoder might not exist 
    # forward is mostly for examining no. parameters
    def forward(self, input: Tensor) -> Tensor:
        embedding = self.encode(input)

        if self.__decoder is None:
            return embedding

        return self.decode(embedding)
        