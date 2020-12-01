# coding = utf-8

from torch import nn, Tensor

from util.conf import Configuration
from model.commons import Squeeze, Reshape


class RNNEncoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(RNNEncoder, self).__init__()

        model_type: str = conf.getHP('encoder')

        if model_type == 'gru':
            model_class = nn.GRU
        elif model_type == 'lstm':
            model_class = nn.LSTM
        else:
            raise ValueError('encoder {:s} is not supported'.format(model_type))

        dim_embedding = conf.getHP('dim_embedding')
        dim_latent = conf.getHP('dim_rnnen_latent')
        bidirectional = conf.getHP('if_rnnen_bidirectional')
        dropout = conf.getHP('rnnen_dropout')
        device = conf.getHP('device')

        self.__transform = model_class(
            input_size = 1,
            hidden_size = dim_latent,
            num_layers = conf.getHP('num_rnnen_layers'),
            batch_first = True,
            bidirectional = bidirectional,
            dropout = dropout
        )
        
        self.__map = model_class(
            input_size = dim_latent * (2 if bidirectional else 1),
            hidden_size = dim_embedding,
            num_layers = 1,
            batch_first = True
        )

        self.__normalize = nn.LayerNorm(dim_embedding, elementwise_affine=False) if conf.getHP('encoder_normalize_embedding') else nn.Identity()

        self.__transform.to(device)
        self.__map.to(device)
        self.__normalize.to(device)


    def forward(self, input: Tensor) -> Tensor:
        output, _ = self.__transform(input)
        output, _ = self.__map(output)
        
        return self.__normalize(output[:, -1])



class RNNDecoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(RNNDecoder, self).__init__()

        model_type: str = conf.getHP('decoder')

        if model_type == 'gru':
            model_class = nn.GRU
        elif model_type == 'lstm':
            model_class = nn.LSTM
        else:
            raise ValueError('decoder {:s} is not supported'.format(model_type))

        dim_embedding = conf.getHP('dim_embedding')
        dim_latent = conf.getHP('dim_rnnde_latent')
        bidirectional = conf.getHP('if_rnnde_bidirectional')
        dropout = conf.getHP('rnnde_dropout')
        device = conf.getHP('device')

        self.dim_series = conf.getHP('dim_series')
            
        self.__transform = model_class(
            input_size = dim_embedding,
            hidden_size = dim_latent,
            num_layers = conf.getHP('num_rnnde_layers'),
            batch_first = True,
            bidirectional = bidirectional, 
            dropout = dropout
        )
        
        self.__map = nn.Sequential(nn.Linear(dim_latent * (2 if bidirectional else 1), dim_latent),
                                   conf.getActivation(conf.getHP('activation_linear')),

                                   nn.Linear(dim_latent, 1, bias=False))
        
        self.__normalize = nn.LayerNorm([self.dim_series, 1], elementwise_affine=False) if conf.getHP('decoder_normalize_reconstruction') else nn.Identity()

        self.__transform.to(device)
        self.__map.to(device)
        self.__normalize.to(device)


    def forward(self, input: Tensor) -> Tensor:    
        input = input.unsqueeze(1).repeat((1, self.dim_series, 1))        

        output, _ = self.__transform(input)
        output = self.__map(output)
                
        return self.__normalize(output.flip(1))
