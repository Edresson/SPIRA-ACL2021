"""
SpiraConv model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish

class SpiraConv(nn.Module):
    def __init__(self, config):
        super(VoiceSplit, self).__init__()
        self.config = config
        self.audio = self.config['audio']
        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(2,2), dilation=(1, 1)),
            nn.BatchNorm2d(32), Mish(), nn.MaxPool2d(2, stride=2),
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(16), Mish(), nn.MaxPool2d(2, stride=2),

            # cnn3
            nn.Conv2d(16, 8, kernel_size=(2, 2), dilation=(2, 1)), 
            nn.BatchNorm2d(8), Mish(), nn.MaxPool2d(2, stride=2),
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 2), dilation=(2, 1)), 
            nn.BatchNorm2d(4), Mish()]

        self.conv = nn.Sequential(*convs)
        # ToDo: Fix input dim
        self.fc = nn.Linear(40, 1)

    def forward(self, x):
        # x: [B, T, num_freq]
        x = self.conv(x)
        print(x.shape)
        x = self.fc(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x