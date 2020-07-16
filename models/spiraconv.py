"""
SpiraConv model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish

class SpiraConv(nn.Module):
    def __init__(self, config):
        super(SpiraConv, self).__init__()
        self.config = config
        self.audio = self.config['audio']

        if self.config.audio['feature'] == 'spectrogram':
            self.num_feature == self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_feature = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_feature = self.config.audio['num_mfcc']

        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7,1), dilation=(2, 1)),
            nn.BatchNorm2d(32), Mish(), nn.MaxPool2d(kernel_size=(2,1)),
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.BatchNorm2d(16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)),

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.BatchNorm2d(8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)),
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)), 
            nn.BatchNorm2d(4), Mish()]

        self.conv = nn.Sequential(*convs)
        # ToDo: Fix input dim
        self.fc1 = nn.Linear(4*self.num_feature, self.config.model['fc1_dim'])
        self.mish = Mish()
        self.fc2 = nn.Linear(self.config.model['fc1_dim'], self.config.model['fc2_dim'])

    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        #print(x.shape)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        #print(x.shape)
        # x: [B, 4, T, num_feature]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 4, num_feature]
        #print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 4*num_feature]
        #print(x.shape)
        x = self.fc1(x) # x: [B, T, fc2_dim]
        #print(x.shape)
        x = self.mish(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        return x