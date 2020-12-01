# coding = utf-8

import os
import struct
import platform
import subprocess
from os.path import isfile
from pathlib import Path
from ctypes import CDLL, c_char_p, c_long
from _ctypes import dlclose

import torch
import numpy as np
from torch.utils.data import Dataset

from util.conf import Configuration
    

class TSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, indices):
        return self.data[indices]



def getSamples(conf: Configuration):
    dim_series = conf.getHP('dim_series')
    size_train = conf.getHP('size_train')
    size_val = conf.getHP('size_val')
    device = conf.getHP('device')

    train_path = conf.getHP('train_path')
    val_path = conf.getHP('val_path')

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_samples = torch.from_numpy(np.fromfile(train_path, dtype=np.float32, count=dim_series * size_train))
        val_samples = torch.from_numpy(np.fromfile(val_path, dtype=np.float32, count=dim_series * size_val))
    else:
        if conf.getHP('sampling_name') == 'coconut' or conf.getHP('sampling_name') == 'uniform':
            train_samples, val_samples = sample(conf)
        else:
            raise ValueError('sampling {:s} is not supported'.format(conf.getHP('sampling_name')))

    if conf.getHP('encoder') == 'gru' or conf.getHP('encoder') == 'lstm':
        train_samples = train_samples.view([-1, dim_series, 1])
        val_samples = val_samples.view([-1, dim_series, 1])
    else:
        train_samples = train_samples.view([-1, 1, dim_series])
        val_samples = val_samples.view([-1, 1, dim_series])

    train_samples = train_samples.to(device)
    val_samples = val_samples.to(device)

    return train_samples, val_samples


# def loadTrainValCocunut(dataset_name, dataset_path, dataset_size, train_size, val_size, series_length=256, sax_length=16, sax_cardinality=8):
def sample(conf: Configuration):
    dataset_path = conf.getHP('database_path')

    train_path = conf.getHP('train_path')
    val_path = conf.getHP('val_path')
    train_indices_path = conf.getHP('train_indices_path')
    val_indices_path = conf.getHP('val_indices_path')

    os.makedirs(Path(train_path).parent, exist_ok=True)
    os.makedirs(Path(val_path).parent, exist_ok=True)
    os.makedirs(Path(train_indices_path).parent, exist_ok=True)
    os.makedirs(Path(val_indices_path).parent, exist_ok=True)

    dim_coconut = conf.getHP('dim_coconut')
    dim_series = conf.getHP('dim_series')
    size_train = conf.getHP('size_train')
    size_val = conf.getHP('size_val')
    size_db = conf.getHP('size_db')

    sampling_method = conf.getHP('sampling_name')

    if sampling_method == 'coconut':
        if not (os.path.exists(train_indices_path) and isfile(train_indices_path)) or not (os.path.exists(val_indices_path) and isfile(val_indices_path)):
            c_functions = CDLL(conf.getHP('coconut_libpath'))

            return_code = c_functions.sample_coconut(c_char_p(dataset_path.encode('ASCII')), 
                                                    c_long(size_db),
                                                    c_char_p(train_indices_path.encode('ASCII')), 
                                                    size_train,
                                                    c_char_p(val_indices_path.encode('ASCII')), 
                                                    size_val, 
                                                    dim_series, 
                                                    conf.getHP('coconut_cardinality'),
                                                    dim_coconut)
            dlclose(c_functions._handle)
            
            if return_code != 0:
                print(return_code)
    elif sampling_method == 'uniform':
        if not (os.path.exists(train_indices_path) and isfile(train_indices_path)) or not (os.path.exists(val_indices_path) and isfile(val_indices_path)):
            train_sample_indices = np.random.randint(0, size_db, size=size_train, dtype=np.int64)
            val_samples_indices = np.random.randint(0, size_db, size=size_val, dtype=np.int64)

            train_sample_indices.tofile(train_indices_path)
            val_samples_indices.tofile(val_indices_path)
    else:
        raise ValueError('sampling {:s} is not supported'.format(sampling_method))

    train_sample_indices = np.fromfile(train_indices_path, dtype=np.int64)
    assert len(train_sample_indices) == size_train
    
    loaded = []
    for index in train_sample_indices:
        sequence = np.fromfile(dataset_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)

        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence) 

    train_samples = np.asarray(loaded, dtype=np.float32)
    train_samples.tofile(train_path)
    train_samples = torch.from_numpy(train_samples)
            
    val_samples_indices = np.fromfile(val_indices_path, dtype=np.int64)
    assert len(val_samples_indices) == size_val

    loaded = []
    for index in val_samples_indices:
        sequence = np.fromfile(dataset_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)

        if not np.isnan(np.sum(sequence)):
            loaded.append(sequence) 

    val_samples = np.asarray(loaded, dtype=np.float32)
    val_samples.tofile(val_path)
    val_samples = torch.from_numpy(val_samples)

    return train_samples, val_samples



class FileContainer(object):
    def __init__(self, filename, binary=True):
        self.filename = filename
        self.binary = binary
        if self.binary:
            self.f = open(filename, "wb")
        else:
            self.f = open(filename, "w")

    def write(self, ts):
        if self.binary:
            s = struct.pack('f' * len(ts), *ts)
            self.f.write(s)
        else:
            self.f.write(" ".join(map(str, ts)) + "\n")

    def close(self):
        self.f.close()



def embedData(model, data_filepath, embedding_filepath, data_size, batch_size = 2000, original_dim = 256, 
              embedded_dim = 16, device = 'cuda', is_rnn = False, encoder = ''):
    if encoder == 'gru' or encoder == 'lstm':
        is_rnn = True

    num_segments = int(data_size / batch_size)

    if data_size < batch_size:
        num_segments = 1
        batch_size = data_size
    else: 
        assert data_size % batch_size == 0

    nan_replacement_original = np.array([0.] * original_dim).reshape([original_dim, 1] if is_rnn else [1, original_dim])
    nan_replacement_embedding = [0.] * embedded_dim

    writer = FileContainer(embedding_filepath)
    
    try:
        with torch.no_grad():
            total_nans = 0

            for segment in range(num_segments):
                batch = np.fromfile(data_filepath, dtype=np.float32, count=original_dim * batch_size, offset=4 * original_dim * batch_size * segment)

                if is_rnn:
                    batch = batch.reshape([-1, original_dim, 1])
                else:
                    batch = batch.reshape([-1, 1, original_dim])

                nan_indices = set()
                for i, sequence in zip(range(batch.shape[0]), batch):
                    if np.isnan(np.sum(sequence)):
                        nan_indices.add(i)
                        batch[i] = nan_replacement_original

                embedding = model.encode(torch.from_numpy(batch).to(device)).detach().cpu().numpy()

                for i in nan_indices:
                    embedding[i] = nan_replacement_embedding

                writer.write(embedding.flatten())
                        
                total_nans += len(nan_indices)

            print('nans = {:d}'.format(total_nans))
    finally:
        writer.close()
