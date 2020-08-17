"""
SpiraConv models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish

class SpiraConvV2(nn.Module):
    ''' Is the same than V1 but we change batchnorm to Group Norm'''
    def __init__(self, config):
        super(SpiraConvV2, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.padding_with_max_lenght = self.config.dataset['padding_with_max_lenght'] or self.config.dataset['split_wav_using_overlapping']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])
        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(2, 1)),
            nn.GroupNorm(16, 32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.GroupNorm(4, 8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)), 
            nn.GroupNorm(2, 4), Mish(), nn.Dropout(p=0.7)]

        self.conv = nn.Sequential(*convs)

        if self.padding_with_max_lenght:
            # its very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer aactivate the network in toy example because is more easy than calculate the conv output
            # get zeros input
            inp = torch.zeros(1, 1, self.max_seq_len, self.num_feature)
            # get out shape
            toy_activation_shape = self.conv(inp).shape
            # set fully connected input dim 
            fc1_input_dim = toy_activation_shape[1]*toy_activation_shape[2]*toy_activation_shape[3]
            self.fc1 = nn.Linear(fc1_input_dim, self.config.model['fc1_dim'])
        else:
            # dinamic calculation num_feature, its useful if you use maxpooling or other pooling in feature dim, and this model dont break
            inp = torch.zeros(1, 1, 500 ,self.num_feature)
            # get out shape 
            self.fc1 = nn.Linear(4*self.conv(inp).shape[-1], self.config.model['fc1_dim'])
        self.mish = Mish()
        self.fc2 = nn.Linear(self.config.model['fc1_dim'], self.config.model['fc2_dim'])
        self.dropout = nn.Dropout(p=0.7)
    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        #print(x.shape)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        #print(x.shape)
        # x: [B, n_filters, T, num_feature]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        #print(x.shape)
        if self.padding_with_max_lenght:
             # x: [B, T*n_filters*num_feature]
            x = x.view(x.size(0), -1)
        else:
             # x: [B, T, n_filters*num_feature]
            x = x.view(x.size(0), x.size(1), -1)
    
       
        #print(x.shape)
        x = self.fc1(x) # x: [B, T, fc2_dim]
        #print(x.shape)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x

class SpiraConvV1(nn.Module):
    def __init__(self, config):
        super(SpiraConvV1, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        self.padding_with_max_lenght = self.config.dataset['padding_with_max_lenght']
        self.max_seq_len = self.config.dataset['max_seq_len']
        print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']
        else:
            self.num_feature = None
            raise ValueError('Feature %s Dont is supported'%self.config.audio['feature'])
        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(2, 1)),
            nn.BatchNorm2d(32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.BatchNorm2d(16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.BatchNorm2d(8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)), 
            nn.BatchNorm2d(4), Mish(), nn.Dropout(p=0.7)]

        self.conv = nn.Sequential(*convs)

        if self.padding_with_max_lenght:
            # its very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer aactivate the network in toy example because is more easy than calculate the conv output
            # get zeros input
            inp = torch.zeros(1, 1, self.max_seq_len, self.num_feature)
            # get out shape
            toy_activation_shape = self.conv(inp).shape
            # set fully connected input dim 
            fc1_input_dim = toy_activation_shape[1]*toy_activation_shape[2]*toy_activation_shape[3]
            self.fc1 = nn.Linear(fc1_input_dim, self.config.model['fc1_dim'])
        else:
            # dinamic calculation num_feature, its useful if you use maxpooling or other pooling in feature dim, and this model dont break
            inp = torch.zeros(1, 1, 500 ,self.num_feature)
            # get out shape 
            self.fc1 = nn.Linear(4*self.conv(inp).shape[-1], self.config.model['fc1_dim'])
        self.mish = Mish()
        self.fc2 = nn.Linear(self.config.model['fc1_dim'], self.config.model['fc2_dim'])
        self.dropout = nn.Dropout(p=0.7)
    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        #print(x.shape)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        #print(x.shape)
        # x: [B, n_filters, T, num_feature]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        #print(x.shape)
        if self.padding_with_max_lenght:
             # x: [B, T*n_filters*num_feature]
            x = x.view(x.size(0), -1)
        else:
             # x: [B, T, n_filters*num_feature]
            x = x.view(x.size(0), x.size(1), -1)
    
       
        #print(x.shape)
        x = self.fc1(x) # x: [B, T, fc2_dim]
        #print(x.shape)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x