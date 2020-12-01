# coding = utf-8

import os
from os.path import basename
import json
import platform
from datetime import date
from typing import Union, List
from pathlib import Path

import torch
from torch import Tensor, nn, optim

from model.activation import LeCunTanh
from model.normalization import AdaNorm


class Configuration:
    def __init__(self, path: str = '', dump: bool = False, existing: bool = False):
        self.defaults = {
            'dataset_name': 'rw',
            'database_path': 'default',
            'query_path': 'default',
            'train_path': 'default',
            'train_indices_path': 'default',
            'val_path': 'default',
            'val_indices_path': 'default',
            'sampling_name': 'coconut',
            'coconut_libpath': 'default',
            'coconut_cardinality': 8,
            'encoder': 'residual',
            'decoder': 'residual',
            'dilation_type': 'exponential',
            'dilation_cons': 1,
            'dilation_base': 1,
            'dilation_slope': 1,
            'dim_series': 256,
            'dim_embedding': 16,
            'dim_en_latent': 256,
            'dim_de_latent': 256,
            'dim_coconut': 16,
            'size_db': 10000000,
            'size_query': 1000,
            'size_train': 200000,
            'size_val': 10000,
            'size_batch': 256,
            'size_kernel': 3,
            'num_en_resblock': 3,
            'num_de_resblock': 2,
            'num_en_channels': 256,
            'num_de_channels': 256,
            'num_epoch': 100,
            'relu_slope': 1e-2,
            'optim_type': 'sgd',
            'momentum': 0.9,
            'lr_mode': 'linear', 
            'lr_cons': 1e-3,
            'lr_max': 1e-3,
            'lr_min': 1e-5,
            'lr_everyk': 2,
            'lr_ebase': 0.9,
            'wd_mode': 'fix', 
            'wd_cons': 1e-4,
            'wd_max': 1e-4,
            'wd_min': 1e-8,
            'device': 'cuda',
            'activation_conv': 'relu',
            'activation_linear': 'lecuntanh',
            'resblock_pre_activation': True,
            'layernorm_type': 'layernorm',
            'layernorm_elementwise_affine': True,
            'adanorm_k': 1 / 10,
            'adanorm_scale': 2.,
            'weightnorm_type': 'none',
            'weightnorm_dim': 0,
            'epsilon': 1e-5,
            'train_type': 'linearlycombine',
            'recons_weight': 1/4,
            'train_detach_query': True,
            'model_init': 'lsuv',
            'lsuv_size': 2000,
            'lsuv_mean': 0,
            'lsuv_std': 1.,
            'lsuv_std_tol': 0.1,
            'lsuv_maxiter': 10,
            'lsuv_ortho': True,
            'torch_rdseed': 1997,
            'cuda_rdseed': 1229,
            'log_folder': 'default',
            'log_filename': 'default',
            'log_filepath': 'default',
            'if_record': True,
            'num_record': 20,
            'record_folder': 'default',
            'orth_regularizer': 'none',
            'srip_mode': 'linear',
            'srip_cons': 0.1,
            'srip_max': 5e-4,
            'srip_min': 0,
            'checkpoint_mode': 'last',
            'checkpoint_k': 1,
            'checkpoint_postfix': 'pickle',
            'checkpoint_folder': 'default',
            'to_embed': True,
            'db_embedding_path': 'default',
            'query_embedding_path': 'default',
            'embed_batch': 2000,
            'encoder_normalize_embedding': True,
            'decoder_normalize_reconstruction': True,
            'dense_growth_rate': 32,
            'dense_bottleneck_multiplier': 4,
            'dense_transition_channels': 256,
            'dense_init_channels': 64,
            'num_en_denselayers': 8,
            'num_en_denseblocks': 7,
            'num_de_denselayers': 8,
            'num_de_denseblocks': 1,
            'default_conf_filename': 'conf.json',
            'reverse_de_dilation': False,
            'num_rnnen_layers': 7,
            'num_rnnde_layers': 7,
            'dim_rnnen_latent': 256,
            'dim_rnnde_latent': 256,
            'if_rnnen_bidirectional': False,
            'if_rnnde_bidirectional': False,
            'rnnen_dropout': 0.4,
            'rnnde_dropout': 0.4,
            'inception_bottleneck_channels': 64,
            'inception_kernel_sizes': [1, 3, 5, 9, 17],
            'num_inceptionen_blocks': 5,
            'num_inceptionde_blocks': 5,
            'to_scale_lc': False,
            'to_scale_lr': False,
            'name': 'default'
        }

        self.legals = {
            'device': {'cpu', 'cuda'},
            'encoder': {'residual', 'dense', 'gru', 'lstm', 'fdj', 'inception'},
            'decoder': {'residual', 'dense', 'singleresidual', 'none', 'gru', 'lstm', 'fdj', 'inception'},
            'activation_conv': {'relu', 'leakyrelu', 'tanh', 'lecuntanh'},
            'activation_linear': {'relu', 'leakyrelu', 'tanh', 'lecuntanh'},
            'layernorm_type': {'layernorm', 'adanorm', 'none'},
            'weightnorm_type': {'weightnorm', 'none'},
            'dilation_type': {'exponential', 'linear', 'fixed'},
            'train_type': {'interleaving', 'linearlycombine'},
            # 'dataset_name': {'rw', 'f5', 'f10', 'f20', 'f30', 'ucr18', 'seismic', 'astro', 'deep1b', 'sald'},
            'sampling_name': {'coconut', 'uniform'},
            'lr_mode': {'linear', 'fix', 'exponentially', 'exponentiallyhalve', 'plateauhalve'},
            'wd_mode': {'linear', 'fix'},
            'srip_mode': {'linear', 'fix'},
            'model_init': {'lsuv', 'default'},
            'orth_regularizer': {'srip', 'none'},
            'checkpoint_mode': {'last', 'everyk', 'none'}
        }

        self.settings = {}

        if path != '':
            if os.path.isdir(path):
                existing = True
                path = os.path.join(path, self.getHP('default_conf_filename'))

            self.loadConf(path, existing)
        
        # if dump and not existing:
        if dump:
            self.dumpConf()


    def getHP(self, name: str):
        if name in self.settings:
            return self.settings[name]
        
        if name in self.defaults:
            return self.defaults[name]
        
        raise ValueError('hyperparmeter {} doesn\'t exist'.format(name))


    def setHP(self, key: str, value):
        self.settings[key] = value


    def __validate(self) -> None:
        for key, value in self.settings.items():
            if key in self.legals and value not in self.legals[key]:
                raise ValueError('illegal setting {} for {} ({})'.format(value, key, ', '.join([str(item) for item in self.legals[key]])))

        assert self.getHP('lr_mode') != 'exponentially' or (0 < self.getHP('lr_ebase') < 1)
        assert (self.getHP('encoder') == 'gru' or self.getHP('encoder') == 'lstm') == (self.getHP('decoder') == 'gru' or self.getHP('decoder') == 'lstm')
        
        if self.getHP('encoder') == 'fdj':
            assert self.getHP('decoder') == 'fdj' or self.getHP('decoder') == 'none'
            assert self.getHP('resblock_pre_activation') == False
        elif self.getHP('decoder') == 'fdj':
            raise ValueError('decoder {:s} shoud have encoder {:s}, while got {:s}'.format(self.getHP('decoder'), 'fdj', self.getHP('encoder')))
        
        if self.getHP('encoder') == 'inception' or self.getHP('decoder') == 'inception':
            inception_kernel_sizes = self.getHP('inception_kernel_sizes')
            assert type(inception_kernel_sizes) == list and len(inception_kernel_sizes) != 0 and 1 in inception_kernel_sizes

        assert self.getHP('database_path') != 'default'
        assert self.getHP('query_path') != 'default'
        assert self.getHP('coconut_libpath') != 'default'


    def __setup(self, existing: bool = False) -> None:
        dataset_name = self.getHP('dataset_name')
        dim_series = self.getHP('dim_series') 

        db_size = self.getHP('size_db')
        assert db_size % 1000000 == 0  

        train_size = self.getHP('size_train')
        assert train_size % 1000 == 0

        encoder_code = self.getHP('encoder')
        decoder_code = self.getHP('decoder')

        sampling_code = self.getHP('sampling_name')
        if sampling_code == 'coconut':
            sampling_code += ('-' + str(self.getHP('dim_coconut')))

        if existing:
            result_root = str(Path(self.getHP('conf_path')).parent)
        else:         
            result_root = os.path.join(os.getcwd(), self.getHP('name'))
            os.makedirs(result_root, exist_ok=True)

        sample_root = os.path.join(result_root, 'samples')
        
        assert self.getHP('database_path') != 'default' and os.path.isfile(self.getHP('database_path'))
        assert self.getHP('query_path') != 'default' and os.path.isfile(self.getHP('query_path'))
        assert self.getHP('coconut_libpath') != 'default' and os.path.isfile(self.getHP('coconut_libpath'))

        if self.getHP('train_path') == 'default':
            filename = '-'.join([sampling_code, dataset_name, str(dim_series), str(int(db_size / 1000000)) + 'm', str(int(train_size / 1000)) + 'k']) + '.bin'
            self.setHP('train_path', os.path.join(sample_root, filename))

        if self.getHP('train_indices_path') == 'default' or (existing and sample_root not in self.getHP('train_indices_path')):
            filename = '-'.join([sampling_code, dataset_name, str(dim_series), str(int(db_size / 1000000)) + 'm-indices', str(int(train_size / 1000)) + 'k']) + '.bin'
            self.setHP('train_indices_path', os.path.join(sample_root, filename))
                
        if self.getHP('val_path') == 'default':
            val_size = self.getHP('size_val')
            # assert val_size % 1000 == 0

            filename = '-'.join([sampling_code, dataset_name, str(dim_series), str(int(db_size / 1000000)) + 'm', str(int(val_size / 1000)) + 'k']) + '.bin'
            self.setHP('val_path', os.path.join(sample_root, filename))

        if self.getHP('val_indices_path') == 'default' or (existing and sample_root not in self.getHP('val_indices_path')):
            val_size = self.getHP('size_val')
            # assert val_size % 1000 == 0

            filename = '-'.join([sampling_code, dataset_name, str(dim_series), str(int(db_size / 1000000)) + 'm-indices', str(int(val_size / 1000)) + 'k']) + '.bin'
            self.setHP('val_indices_path', os.path.join(sample_root, filename))

        embedding_prefix = '-'.join([dataset_name, str(dim_series), str(int(db_size / 1000000)) + 'm',
                                     encoder_code, decoder_code, str(self.getHP('dim_embedding')),
                                     sampling_code, str(int(train_size / 1000)) + 'k'])

        if self.getHP('db_embedding_path') == 'default':
            self.setHP('db_embedding_path', os.path.join(result_root, embedding_prefix + '.bin'))

        if self.getHP('query_embedding_path') == 'default':
            # assert self.getHP('size_query') % 1000 == 0
            self.setHP('query_embedding_path', os.path.join(result_root, embedding_prefix + '-1k.bin'))

        if self.getHP('log_filepath') == 'default':
            log_folder = self.getHP('log_folder')

            if log_folder == 'default':
                log_folder = result_root
                
            log_filename = self.getHP('log_filename')
            
            if log_filename == 'default':
                log_filename = 'fit.log'
    
            log_filepath = os.path.join(log_folder, log_filename)
            
            self.setHP('log_filepath', log_filepath)

        if self.getHP('record_folder') == 'default':
            self.setHP('record_folder', result_root)
    
        if self.getHP('checkpoint_folder') == 'default':
            self.setHP('checkpoint_folder', result_root)

        self.default_confpath = os.path.join(result_root, self.getHP('default_conf_filename'))


    def loadConf(self, path: str, existing: bool = False) -> None:
        with open(path, 'r') as fin:
            loaded = json.load(fin)

            if 'settings' in loaded:
                self.settings = loaded['settings']

                local_defaults = self.defaults
                self.defaults = loaded['defaults']
                
                for name, value in local_defaults.items():
                    if name not in self.defaults:
                        self.defaults[name] = value
            else:
                self.settings = loaded

        self.settings['conf_path'] = path

        self.__validate()
        self.__setup(existing)


    def dumpConf(self, path: str = None) -> None:
        with open(self.default_confpath if path is None else path, 'w') as fout:
            json.dump({'settings': self.settings, 'defaults': self.defaults}, fout, sort_keys=True, indent=4)


    # TODO this design (of getActivation in conf) is a little tricky
    def getActivation(self, name: str) -> nn.Module:
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'lecuntanh':
            return LeCunTanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'leakyrelu':
            return nn.LeakyReLU(self.getHP('relu_slope'))

        return nn.Identity()


    def getWeightNorm(self, model: nn.Module) -> nn.Module:
        weightnorm_type = self.getHP('weightnorm_type')

        if weightnorm_type == 'weightnorm' or self.getHP('encoder') == 'fdj':
            return nn.utils.weight_norm(model, dim=self.getHP('weightnorm_dim'))

        return model


    def getLayerNorm(self, shape: Union[int, List[int], torch.Size]) -> nn.Module:
        if self.getHP('encoder') != 'fdj':
            layernorm_type = self.getHP('layernorm_type')
            if layernorm_type == 'layernorm':
                return nn.LayerNorm(shape, elementwise_affine=self.getHP('layernorm_elementwise_affine'))
            elif layernorm_type == 'adanorm':
                return AdaNorm(shape, self.getHP('adanorm_k'), self.getHP('adanorm_scale'), self.getHP('eps'), self.getHP('layernorm_elementwise_affine'))
        
        return nn.Identity()

    
    # depth starts from 1
    def getDilatoin(self, depth: int, to_encode: bool = True) -> int:
        dilation_type = self.getHP('dilation_type')

        if not to_encode:
            decoder_name = self.getHP('decoder')

            if decoder_name == 'residual' or decoder_name == 'fdj':
                depth = self.getHP('num_de_resblock') + 1 - depth
            elif decoder_name == 'dense':
                depth = self.getHP('num_de_denseblocks') + 1 - depth
            elif not dilation_type == 'fixed':
                raise ValueError('illegal decoder {:s} to change dilation (residual/dense only)'.format(self.getHP('decoder')))

        if dilation_type == 'exponential':
            return int(2 ** (depth - 1))
        elif dilation_type == 'linear':
            return self.getHP('dilation_base') + self.getHP('dilation_slope') * (depth - 1)
        
        return self.getHP('dilation_constant')
    
