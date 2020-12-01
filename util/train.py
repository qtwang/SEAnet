#coding=utf-8

import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from ray import tune
from apex import amp

from model.loss import L2ScaledTransformation, L2Rreconstruction
from util.data import TSDataset


class AlternativeTrainingAutoencoder(tune.Trainable):
    # def setup(self, config):
    def _setup(self, config):
        torch.manual_seed(97)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(29)
        
        self.epoch = 0
        self.ALTERNATIVE_EPOCHES = int(config["epoch"] / 5 * 4)
        self.ADJUSTION_EPOCHES = int(config["epoch"] / 4)

        self.train_dataset = TSDataset(config["train_samples"].cuda())
        self.train_db_loader = DataLoader(self.train_dataset, batch_size=config.get('batch_size', 64), shuffle=True)
        self.train_query_loader = DataLoader(self.train_dataset, batch_size=config.get('batch_size', 64), shuffle=True)

        self.val_dataset = TSDataset(config["val_samples"].cuda())
        self.val_db_loader = DataLoader(self.val_dataset, batch_size=config.get('batch_size', 64), shuffle=True)
        self.val_query_loader = DataLoader(self.val_dataset, batch_size=config.get('batch_size', 64), shuffle=True)
        
        self.transformation_loss = L2ScaledTransformation().cuda()
        self.reconstruction_loss = L2Rreconstruction().cuda()

        if 'negative_slope' in config:
            self.model = config["model"](dim_embedding=config.get('dim_embedding', 16),
                                         dim_sequence=config.get('dim_sequence', 256),
                                         negative_slope=config.get('negative_slope', 1e-2)).cuda()
        else:
            self.model = config["model"](dim_embedding=config.get('dim_embedding', 16),
                                         dim_sequence=config.get('dim_sequence', 256)).cuda()

        self.optimizer = optim.SGD(self.model.parameters(), 
                                   lr=config.get('lr', 1e-3), 
                                   momentum=config.get('momentum', 0.9), 
                                   weight_decay=config.get('weight_decay', 1e-5))
                                   
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=config.get('opt_level', 'O0'))

         
    # def step(self):
    def _train(self):   
        self.__adjust_learning_rate()
        self.epoch += 1

        train_reconstruction_batch = []
        if self.epoch < self.ALTERNATIVE_EPOCHES:
            for batch in self.train_db_loader:
                self.optimizer.zero_grad()

                reconstruction_error = self.reconstruction_loss(batch, self.model(batch))
                
                # reconstruction_error.backward()
                with amp.scale_loss(reconstruction_error, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer.step()

                train_reconstruction_batch.append(reconstruction_error.detach().item())
        else:
            with torch.no_grad():
                for batch in self.train_db_loader:
                    train_reconstruction_batch.append(self.reconstruction_loss(batch, self.model(batch)).detach().item())

        train_diffences_batch = []
        for db_batch, query_batch in zip(self.train_db_loader, self.train_query_loader):
            self.optimizer.zero_grad()
            
            transformation_error = self.transformation_loss(db_batch, query_batch, self.model.encode(db_batch), self.model.encode(query_batch))

            # transformation_error.backward()
            with amp.scale_loss(transformation_error, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.optimizer.step()
            
            train_diffences_batch.append(transformation_error.detach().item())
        
        val_diffences_batch = []
        with torch.no_grad():
            for db_batch, query_batch in zip(self.val_db_loader, self.val_query_loader):
                val_diffences_batch.append(self.transformation_loss(db_batch, query_batch, self.model.encode(db_batch), self.model.encode(query_batch)).detach().item())                
        
        return {'val_diff': np.mean(val_diffences_batch), 'train_diff': np.mean(train_diffences_batch), 'rec_error': np.mean(train_reconstruction_batch)}


    def __adjust_learning_rate(self):
        if self.epoch % self.ADJUSTION_EPOCHES == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1


    # def save_checkpoint(self, checkpoint_dir):
    def _save(self, checkpoint_dir):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': amp.state_dict()
        }
        
        # checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        checkpoint_path = os.path.join(checkpoint_dir, 'amp_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path


    # def load_checkpoint(self, checkpoint_path):
    def _restore(self, checkpoint_path):
        # self.model.load_state_dict(torch.load(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])

        
    # def cleanup(self):
    def _stop(self):
        self.train_dataset = None
        self.train_db_loader = None
        self.train_query_loader = None
        
        self.val_dataset = None
        self.val_db_loader = None
        self.val_query_loader = None
        
        self.model = None
        self.transformation_loss = None
        self.reconstruction_loss = None
        self.optimizer = None
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print('Welcome to where the training methods got defined!')
