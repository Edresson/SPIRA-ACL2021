import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import stack
import numpy as np
import pandas as pd
import random
from torch.nn.utils.rnn import pad_sequence
import torchaudio
class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """
    def __init__(self, c, ap, train=True):
        # set random seed
        random.seed(c['seed'])
        self.c = c
        self.ap = ap
        self.train = train
        self.dataset_csv = c.dataset['train_csv'] if train else c.dataset['test_csv']
        self.dataset_root = c.dataset['train_data_root_path'] if train else c.dataset['test_data_root_path']
        self.noise_csv = c.dataset['noise_csv'] 
        self.noise_root = c.dataset['noise_data_root_path']
        assert os.path.isfile(self.dataset_csv),"Test or Train CSV file don't exists! Fix it in config.json"
        assert os.path.isfile(self.noise_csv),"Noise CSV file don't exists! Fix it in config.json"
        
        # read csvs
        self.dataset_list = pd.read_csv(self.dataset_csv, sep=',').values
        self.noise_list = pd.read_csv(self.noise_csv, sep=',').values
        # noise config
        self.num_noise_files = len(self.noise_list)-1
        self.control_class = c.dataset['control_class']
        self.patient_class = c.dataset['patient_class']

    def __getitem__(self, idx):
        wav = self.ap.load_wav(os.path.join(self.dataset_root, self.dataset_list[idx][0]))
        class_name = self.dataset_list[idx][1]

        # its assume that noise file is biggest than wav file !!
        if self.c.data_aumentation['insert_noise']:
            if self.control_class == class_name: # if sample is a control sample
                #print('antes',wav.shape)
                # torchaudio.save('antes_control.wav', wav, self.ap.sample_rate)
                for _ in range(self.c.data_aumentation['num_noise_control']):
                    # choise random noise file
                    noise_wav = self.ap.load_wav(os.path.join(self.noise_root, self.noise_list[random.randint(0, self.num_noise_files)][0]))
                    noise_wav_len = noise_wav.shape[1]
                    wav_len = wav.shape[1]
                    noise_start_slice = random.randint(0,noise_wav_len-(wav_len+1))
                    # sum two diferents noise
                    noise_wav = noise_wav[:,noise_start_slice:noise_start_slice+wav_len]
                    # get random max amp for noise
                    max_amp = random.uniform(self.c.data_aumentation['noise_min_amp'], self.c.data_aumentation['noise_max_amp'])
                    reduct_factor = max_amp/float(noise_wav.max().numpy())
                    noise_wav = noise_wav*reduct_factor
                    wav = wav + noise_wav
                #torchaudio.save('depois_controle.wav', wav, self.ap.sample_rate)
                
            elif self.patient_class == class_name: # if sample is a patient sample
                for _ in range(self.c.data_aumentation['num_noise_patient']):
                    
                    # torchaudio.save('antes_patiente.wav', wav, self.ap.sample_rate)
                    # choise random noise file
                    noise_wav = self.ap.load_wav(os.path.join(self.noise_root, self.noise_list[random.randint(0, self.num_noise_files)][0]))
                    noise_wav_len = noise_wav.shape[1]
                    wav_len = wav.shape[1]
                    noise_start_slice = random.randint(0,noise_wav_len-(wav_len+1))
                    # sum two diferents noise
                    noise_wav = noise_wav[:,noise_start_slice:noise_start_slice+wav_len]
                    # get random max amp for noise
                    max_amp = random.uniform(self.c.data_aumentation['noise_min_amp'], self.c.data_aumentation['noise_max_amp'])
                    reduct_factor = max_amp/float(noise_wav.max().numpy())
                    noise_wav = noise_wav*reduct_factor
                    wav = wav + noise_wav
                
                #torchaudio.save('depois_patient.wav', wav, self.ap.sample_rate)
                
        # feature shape (Batch_size, n_features, timestamp)
        feature = self.ap.get_feature_from_audio(wav)
        # transpose for (Batch_size, timestamp, n_features)
        feature = feature.transpose(1,2)
        # remove batch dim = (timestamp, n_features)
        feature = feature.reshape(feature.shape[1:])
        # generate tensor with zeros for each timestep
        target = torch.zeros(feature.shape[0],1)+class_name
        return feature, target

    def __len__(self):
        return len(self.dataset_list)

def train_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=True),
                          batch_size=c.train_config['batch_size'],
                          shuffle=True,
                          num_workers=c.train_config['num_workers'],
                          collate_fn=own_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)

def eval_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=False),
                          collate_fn=own_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])

def own_collate_fn(batch):
    features = []
    targets = []
    for feature, target in batch:
        features.append(feature)
        targets.append(target)
    # padding with zeros timestamp dim
    features = pad_sequence(features, batch_first=True, padding_value=0)

    # its padding with zeros but mybe its a problem because 
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # list to tensor
    #targets = stack(targets, dim=0)
    return features, targets