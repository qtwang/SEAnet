# coding = utf-8

import os
import io
import logging  
from datetime import date
from timeit import default_timer as timer

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model.initialization import LSUVinit
from model.normalization import getSRIPTerm
from model.loss import ScaledL2Recons, ScaledL2Trans
from model.builder import AEBuilder
from util.data import TSDataset, getSamples
from util.conf import Configuration
from util.data import embedData


class Experiment:
    def __init__(self, conf: Configuration):
        self.__conf = conf
        self.has_setup = False
        self.epoch = 0

        self.device = conf.getHP('device')

        self.max_epoch = self.__conf.getHP('num_epoch')

        self.checkpoint_folder = conf.getHP('checkpoint_folder')
        self.checkpoint_postfix = conf.getHP('checkpoint_postfix')


    def loadModel(self, epoch: int = -1) -> nn.Module:
        assert epoch < self.max_epoch
        checkpoint_path = os.path.join(self.checkpoint_folder, str(epoch) + '.' + self.checkpoint_postfix)

        if epoch == -1:
            epoch = self.max_epoch
            checkpoint_path = os.path.join(self.checkpoint_folder, str(epoch) + '.' + self.checkpoint_postfix)

            while not os.path.exists(checkpoint_path) and epoch >= 0:
                epoch -= 1
                checkpoint_path = os.path.join(self.checkpoint_folder, str(epoch) + '.' + self.checkpoint_postfix)

            if epoch != -1:
                print('loading checkpoint at epoch {:d}'.format(epoch))
        else:
            assert os.path.exists(checkpoint_path)

        if epoch == -1:
            return None

        self.epoch = epoch

        model = AEBuilder(self.__conf)
        with open(checkpoint_path, 'rb') as fin:
            model.load_state_dict(torch.load(fin))

        return model


    def getSample(self, size: int = -1) -> torch.Tensor:
        _, val_samples = getSamples(self.__conf)

        if size == -1:
            return val_samples
        elif size <= self.__conf.getHP('size_val'):
            indices = torch.randperm(val_samples.shape[0])
            return val_samples[indices][: size].to(self.device)
        else:
            raise ValueError('cannot provide {:d} samples ({:d} valset)'.format(size, self.__conf.getHP('size_val')))


    def setup(self) -> None:
        self.has_setup = True

        logging.basicConfig(filename=self.__conf.getHP('log_filepath'), 
                            filemode='a+', 
                            format='%(asctime)s,%(msecs)d %(levelname).3s [%(filename)s:%(lineno)d] %(message)s', 
                            level=logging.DEBUG,
                            datefmt='%m/%d/%Y:%I:%M:%S')
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        torch.manual_seed(self.__conf.getHP('torch_rdseed'))
        if self.device == 'cuda':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.manual_seed_all(self.__conf.getHP('cuda_rdseed'))
            else:
                raise ValueError('cuda is not available')

        batch_size = self.__conf.getHP('size_batch')

        train_samples, val_samples = getSamples(self.__conf)

        self.train_db_loader = DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=True)
        self.train_query_loader = DataLoader(TSDataset(train_samples), batch_size=batch_size, shuffle=True)

        self.val_db_loader = DataLoader(TSDataset(val_samples), batch_size=batch_size, shuffle=True)
        self.val_query_loader = DataLoader(TSDataset(val_samples), batch_size=batch_size, shuffle=True)

        dim_series = self.__conf.getHP('dim_series')
        dim_embedding = self.__conf.getHP('dim_embedding')

        if self.__conf.getHP('to_scale_lc'):
            self.trans_loss = ScaledL2Trans(dim_series, dim_embedding, to_scale=True).to(self.device)
        else:
            self.trans_loss = ScaledL2Trans().to(self.device)

        if self.__conf.getHP('to_scale_lr'):
            self.recons_reg = ScaledL2Recons(dim_series, to_scale=True).to(self.device)
        else:
            self.recons_reg = ScaledL2Recons().to(self.device)

        self.model = self.loadModel()
        if self.model is None:
            self.model = AEBuilder(self.__conf)
            self.model = self.__init_model(self.model, val_samples)

        self.optimizer = self.__getOptimizer()

        self.checkpoint_mode = self.__conf.getHP('checkpoint_mode')
        if self.checkpoint_mode != 'none':
            if self.checkpoint_mode == 'everyk':
                self.checkpoint_k = self.__conf.getHP('checkpoint_k')

        if self.__conf.getHP('if_record'):
            indices = torch.randperm(val_samples.shape[0])

            self.samples2plot = val_samples[indices][: self.__conf.getHP('num_record')].to(self.device)
            self.record_folder = self.__conf.getHP('record_folder')

        self.detch_query = self.__conf.getHP('train_detach_query')

        self.encoder_only = self.__conf.getHP('decoder') == 'none'
        if not self.encoder_only:
            self.recons_weight = self.__conf.getHP('recons_weight')

        self.orth_regularizer = self.__conf.getHP('orth_regularizer')
        if self.orth_regularizer == 'srip':
            if self.__conf.getHP('srip_mode') == 'fix':
                self.srip_weight = self.__conf.getHP('srip_cons')
            elif self.__conf.getHP('srip_mode') == 'linear':
                self.srip_weight = self.__conf.getHP('srip_max')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            

    def run(self) -> None:
        if not self.has_setup:
            self.setup()

        self.__checkpoint(persist_model=False)

        while self.epoch < self.max_epoch:
            start = timer()

            self.__adjust_lr()
            self.__adjust_wd()
            if self.orth_regularizer == 'srip':
                self.__adjust_srip()

            self.epoch += 1

            self.__train()
            self.__validate()

            self.logger.info('e{:d} time = {:.3f}s'.format(self.epoch, timer() - start))

            self.__checkpoint()

        if self.__conf.getHP('to_embed'):
            embedData(self.model, self.__conf.getHP('database_path'), self.__conf.getHP('db_embedding_path'), 
                      self.__conf.getHP('size_db'), batch_size=self.__conf.getHP('embed_batch'), original_dim=self.__conf.getHP('dim_series'), 
                      embedded_dim=self.__conf.getHP('dim_embedding'), device=self.device, encoder=self.__conf.getHP('encoder'))    
            embedData(self.model, self.__conf.getHP('query_path'), self.__conf.getHP('query_embedding_path'), 
                      self.__conf.getHP('size_query'), batch_size=self.__conf.getHP('embed_batch'), original_dim=self.__conf.getHP('dim_series'), 
                      embedded_dim=self.__conf.getHP('dim_embedding'), device=self.device, encoder=self.__conf.getHP('encoder'))        


    def __train(self) -> None:
        recons_errors = []
        orth_terms = []
        trans_errors = []

        if self.__conf.getHP('train_type') == 'interleaving':
            if not self.encoder_only:
                for batch in self.train_db_loader:
                    self.optimizer.zero_grad()

                    embedding = self.model.encode(batch)
                    reconstructed = self.model.decode(embedding)

                    recons_term = self.recons_weight * self.recons_reg(batch, reconstructed)
                    orth_term = self.__orth_reg()
                    regularization = recons_term + orth_term

                    regularization.backward()
                    self.optimizer.step()

                    recons_errors.append(recons_term.detach().item())
                    orth_terms.append(orth_term.detach().item())

                self.logger.info('t{:d} recons = {:.4f}'.format(self.epoch, np.mean(recons_errors)))
                self.logger.info('t{:d} orth = {:.4f}'.format(self.epoch, np.mean(orth_terms)))

            for db_batch, query_batch in zip(self.train_db_loader, self.train_query_loader):
                self.optimizer.zero_grad()

                if self.detch_query:
                    with torch.no_grad():
                        query_embedding = self.model.encode(query_batch).detach()
                        query_batch = query_batch.detach()
                else:
                    query_embedding = self.model.encode(query_batch)
                
                db_embedding = self.model.encode(db_batch)
                trans_error = self.trans_loss(db_batch, query_batch, db_embedding, query_embedding)
                
                trans_error.backward()
                self.optimizer.step()

                trans_errors.append(trans_error.detach().item())
                
            logging.info('t{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))

        elif self.__conf.getHP('train_type') == 'linearlycombine':
            for db_batch, query_batch in zip(self.train_db_loader, self.train_query_loader):
                self.optimizer.zero_grad()

                if self.detch_query:
                    with torch.no_grad():
                        query_embedding = self.model.encode(query_batch).detach()
                        query_batch = query_batch.detach()
                else:
                    query_embedding = self.model.encode(query_batch)
                
                db_embedding = self.model.encode(db_batch)
                
                if not self.encoder_only:
                    db_reconstructed = self.model.decode(db_embedding)
                    recons_term = self.recons_weight * self.recons_reg(db_batch, db_reconstructed)
                else:
                    recons_term = torch.zeros(1).to(self.device)

                trans_error = self.trans_loss(db_batch, query_batch, db_embedding, query_embedding)
                orth_term = self.__orth_reg()
                
                loss = trans_error + recons_term + orth_term

                loss.backward()
                self.optimizer.step()

                recons_errors.append(recons_term.detach().item())
                orth_terms.append(orth_term.detach().item())
                trans_errors.append(trans_error.detach().item())

            self.logger.info('t{:d} recons = {:.4f}'.format(self.epoch, np.mean(recons_errors)))
            self.logger.info('t{:d} orth = {:.4f}'.format(self.epoch, np.mean(orth_terms)))
            self.logger.info('t{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))
                
        else:
            raise ValueError('cannot train')


    def __validate(self) -> None:
        trans_errors = []

        with torch.no_grad():
            for db_batch, query_batch in zip(self.val_db_loader, self.val_query_loader): 
                db_embedding = self.model.encode(db_batch)
                query_embedding = self.model.encode(query_batch)
                
                trans_error = self.trans_loss(db_batch, query_batch, db_embedding, query_embedding)
                trans_errors.append(trans_error.detach().item())                

        self.logger.info('v{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))


    def __checkpoint(self, persist_model: bool = True) -> None:
        if self.__conf.getHP('if_record'):
            if self.encoder_only:
                fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            else:
                fig, ax = plt.subplots(3, 1, figsize=(12, 9))
            
            with torch.no_grad():
                for series in torch.squeeze(self.samples2plot).detach().cpu():
                    ax[0].plot(series)

                embedding = self.model.encode(self.samples2plot)
                for series in embedding.detach().cpu():
                    ax[1].plot(series)

                if not self.encoder_only:
                    reconstructed = self.model.decode(embedding)
                    for series in torch.squeeze(reconstructed).detach().cpu():
                        ax[2].plot(series)   

            fig.tight_layout()
            plt.savefig(self.record_folder + str(self.epoch) + '.eps', dpi=456)
        
        if persist_model and self.checkpoint_mode != 'none' and (self.epoch == self.max_epoch or (self.checkpoint_mode == 'everyk' and self.epoch % self.checkpoint_k == 0)):
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, str(self.epoch) + '.' + self.checkpoint_postfix))


    def __orth_reg(self) -> torch.Tensor:
        if self.orth_regularizer == 'srip':
            return self.srip_weight * getSRIPTerm(self.model, self.device)

        return torch.zeros(1).to(self.device)


    def __adjust_lr(self) -> None:
        # should be based on self.epoch and hyperparameters ONLY for easily resumming

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            break
        
        new_lr = current_lr

        if self.__conf.getHP('lr_mode') == 'linear':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')

            new_lr = lr_max - self.epoch * (lr_max - lr_min) / self.max_epoch
        elif self.__conf.getHP('lr_mode') == 'exponentiallyhalve':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')
            
            for i in range(1, 11):
                if (self.max_epoch - self.epoch) * (2 ** i) == self.max_epoch:
                    new_lr = lr_max / (10 ** i)
                    break

            if new_lr < lr_min:
                new_lr = lr_min
        elif self.__conf.getHP('lr_mode') == 'exponentially':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')
            lr_k = self.__conf.getHP('lr_everyk')
            lr_ebase = self.__conf.getHP('lr_ebase')

            lr_e = int(np.floor(self.epoch / lr_k))
            new_lr = lr_max * (lr_ebase ** lr_e)

            if new_lr < lr_min:
                new_lr = lr_min
        elif self.__conf.getHP('lr_mode') == 'plateauhalve':
            raise ValueError('plateauhalve is not yet supported')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


    def __adjust_wd(self):
        # should be based on self.epoch and hyperparameters ONLY for easily resumming

        for param_group in self.optimizer.param_groups:
            current_wd = param_group['weight_decay']
            break
        
        new_wd = current_wd

        if self.__conf.getHP('wd_mode') == 'linear':
            wd_max = self.__conf.getHP('wd_max')
            wd_min = self.__conf.getHP('wd_min')

            new_wd = wd_min + self.epoch * (wd_max - wd_min) / self.max_epoch

        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = new_wd


    def __adjust_srip(self):
        # should be based on self.epoch and hyperparameters ONLY for easily resumming
        
        if self.__conf.getHP('srip_mode') == 'linear':
            srip_max = self.__conf.getHP('srip_max')
            srip_min = self.__conf.getHP('srip_min')

            self.srip_weight = srip_max - self.epoch * (srip_max - srip_min) / self.max_epoch



    def __getOptimizer(self) -> optim.Optimizer:
        if self.__conf.getHP('optim_type') == 'sgd':
            if self.__conf.getHP('lr_mode') == 'fix':
                initial_lr = self.__conf.getHP('lr_cons')
            else:
                initial_lr = self.__conf.getHP('lr_max')

            if self.__conf.getHP('wd_mode') == 'fix':
                initial_wd = self.__conf.getHP('wd_cons')
            else:
                initial_wd = self.__conf.getHP('wd_min')

            momentum = self.__conf.getHP('momentum')

            return optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=initial_wd)

        raise ValueError('cannot obtain optimizer')


    def __init_model(self, model: nn.Module, samples: torch.Tensor = None) -> nn.Module:
        if self.__conf.getHP('model_init') == 'lsuv':
            assert samples is not None

            return LSUVinit(model, samples[torch.randperm(samples.shape[0])][: self.__conf.getHP('lsuv_size')], 
                            needed_mean=self.__conf.getHP('lsuv_mean'), needed_std=self.__conf.getHP('lsuv_std'), 
                            std_tol=self.__conf.getHP('lsuv_std_tol'), max_attempts=self.__conf.getHP('lsuv_maxiter'), 
                            do_orthonorm=self.__conf.getHP('lsuv_ortho'))

        return model
