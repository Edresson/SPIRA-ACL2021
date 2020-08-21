"""
SpiraConvLSTM models
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish
from utils.conv_lstm import utfpr_convLSTM_v1, weights_init

class UTF_SPIRA_ConvLSTM_v1(nn.Module):
    def __init__(self, config):
        super(UTF_SPIRA_ConvLSTM_v1, self).__init__()        
        self.config                  = config
        self.audio                   = self.config['audio']
        self.padding_with_max_lenght = self.config.dataset['padding_with_max_lenght']
        self.max_seq_len             = self.config.dataset['max_seq_len']
        self.sequence_step_size      = self.config.model['sequence_step_size']        # MFCC x axis interval to be analyzed per time step in the convLSTM
        self.input_channels          = self.config.model['input_channels']            # 1, as we are analysing an image (MFCC) which is a tensor with size [sequence_step_size, num_features]
        self.num_filters             = self.config.model['conv_lstm_num_filters']
        self.filter_size             = self.config.model['conv_lstm_filter_size']     # TODO: use different filter sizes for different cells
        self.padding                 = int((self.filter_size - 1) / 2)                # guarantees output dim = input dim
        #print("Model Inicialized With Feature %s "%self.config.audio['feature'])
        
        if self.config.audio['feature'] == 'spectrogram':
            self.num_features = self.config.audio['num_freq']
        elif self.config.audio['feature'] == 'melspectrogram':
            self.num_features = self.config.audio['num_mels']
        elif self.config.audio['feature'] == 'mfcc':
            self.num_features = self.config.audio['num_mfcc']
        else:
            self.num_features = None
            raise ValueError('Feature %s is not supported'%self.config.audio['feature'])
           
        self.conv                    = nn.Conv2d(self.input_channels + self.num_features, 4 * self.num_features, self.filter_size, 1, self.padding) # 2 input channels: input and hidden state
        
        if self.padding_with_max_lenght:
            # it's very useful because if you change the convlutional arquiture the model calculate its, and you dont need change this :)
            # I prefer aactivate the network in toy example because is more easy than calculate the conv output
            #print('max_seq_len: ', self.max_seq_len)
            #print('num_features: ', self.num_features)
            # inp                  = torch.zeros(1, 1, self.max_seq_len, self.num_features) # get zeros input
            inp                  = torch.zeros(1,
                                               self.num_filters + self.input_channels,
                                               self.sequence_step_size,
                                               self.num_features) 
            # toy_activation_shape = self.conv(inp).shape # get out shape
            # fc1_input_dim        = toy_activation_shape[1]*toy_activation_shape[2]*toy_activation_shape[3]
            fc1_input_dim = self.num_filters*self.num_features
            self.fc1             = nn.Linear(fc1_input_dim, self.config.model['fc1_dim'])
        else:
            inp      = torch.zeros(1, self.num_filters + self.input_channels, self.sequence_step_size, self.num_features) # (batch_size, n_filters, seq_len, num_features) # dinamic calculation num_features, its useful if you use maxpooling or other pooling in feature dim, and this model dont break # n_filters initially set to 2 due to the concatenation of the current input with LSTM hidden state
            self.fc1 = nn.Linear(inp.size(3) * self.num_filters, self.config.model['fc1_dim']) # get output shape 
            #self.fc1 = nn.Linear(4*self.conv(inp).shape[-1], self.config.model['fc1_dim'])
            
        self.mish                    = Mish()
        self.fc2                     = nn.Linear(self.config.model['fc1_dim'], self.config.model['fc2_dim'])
        self.dropout                 = nn.Dropout(p=0.7)
        
                                            #( shape   , inp_channels       , num_filters     , filter_size     , num_featurees    , num_layers, sequence )
        self.convLSTM                = utfpr_convLSTM_v1(inp.shape, self.input_channels, self.num_filters, self.filter_size, self.num_features, 3         , self.sequence_step_size )
        self.convLSTM.apply(weights_init)
        self.convLSTM.cuda()
        
        
        #print('##########################################################\n')
        #print('ConvLSTM further details: \n')
        #print('self.num_filters:           ', self.num_filters)
        #print('convlstm module:            ', self.convLSTM)
        #print('params:                     ')
        self.params       = self.convLSTM.parameters()
        #for p in self.params:
           #print('param                    ', p.size())
           #print('mean ', torch.mean(p))
        #print('\n##########################################################\n')
        
        
    def forward(self, x):
        #print('\n############################## Inside spiraconv.py ##############################')
        # x: [batch_size, timestamp, num_features]               
        #print('x.shape:                          ', x.size())
        x = x.unsqueeze(1) # -> [batch_size, 1, timestamp, num_features]
        #print('x.shape:                          ', x.size())
        
        initial_hidden_state = self.convLSTM.init_hidden_states(x.size(0))                     
        
        #print('len(initial_hidden_state):        ', len(initial_hidden_state))
        #print('initial_hidden_state[0][0].size():', initial_hidden_state[0][0].size())
        #print('initial_hidden_state is a tuple with 2 tensors')
        
        
        x = self.convLSTM(x, initial_hidden_state)                                                   # x after convLSTM -> tuple(hidden_states, output)
        
        #print('\n############################## Inside spiraconv.py ##############################') # back from utfpr_convLSTM_v1.py
        # x is a tuple of tensors (hidden_state, output). hidden_state is a list of 
        #print('type(x):                          ', type(x))                                         # tuple with shape (list_hidden_states, output_tensor)                              
        #print('type(x[0]):                       ', type(x[0]))                                      # list of hidden_states                                  
        #print('len(x[0]):                        ', len(x[0]))                                       # 1 convLSTM layer produces length 2, as it creates 2 convLSTM cells by default (TODO: change that layer, just realized that's not how things should be)
        #print('type(x[1]):                       ', type(x[1]))                                      # output tensor from convLSTM. It is the concatenation of the MFCC with the unpadded hidden_states (simply cut off the end of the input, which were originally padded with zeros) 
        #print('x[1].size():                      ', x[1].size())
        #print('type(x[0][0]):                    ', type(x[0][0]))                                    
        #print('len(x[0]):                        ', len(x[0][0]))                                    
        #print('type(x[0][0][0]):                 ', type(x[0][0][0]))                                
        #print('x[0][0][0].size():                ', x[0][0][0].size())                               
        
        #x = x.transpose(1, 2).contiguous()                                                           # -> [batch_size, timestamp, n_filters, num_features] (comment needs revision)
        #print('after transpose: ', x.shape)
        
        if self.padding_with_max_lenght:
            #print('before x[1].view(x[1].size(1), x[1].size(0) * x[1].size(2)) ', x[1].size())
            x = x[1].contiguous()
            #print('x[1].size() after  x[1].contiguous():                       ', x[1].size())
            #print('x.size()                                                    ', x.size())
            x = x[1].view(x[1].size(1), x[1].size(0) * x[1].size(2)) # -> [batch_size, timestamp * n_filters * num_features] (comment needs revision)
            #print('after  x[1].view(x[1].size(1), x[1].size(0) * x[1].size(2)) ', x.size())
        else:
            #print('before x[1].view(x[1].size(0), x[1].size(1), -1): ', x[1].size())
            x = x[1].contiguous()
            #print('after  x[1].contiguous():                         ', x[1].size())
            x = x[1].view(x[1].size(2), x[1].size(0), -1) # -> [batch_size, timestamp, num_features * num_filters] (comment needs revision)
            #print('after  x[1].view(x[1].size(0), x[1].size(1), -1): ', x.size())
        
        #print('x.size():                         ', x.size())
        #print('x[0, :].size():                         ', x[0, :].size())
        #print(self.fc1.weight.size())
        x = self.fc1(x)                                                                               # x: (timestamp, n_filters, fc2_dim)
        #print('after fully connected 1:          ', x.size())
        
        x = self.mish(x)
        
        x = self.dropout(x)
        #print('after dropout:                    ', x.size())
        
        x = self.fc2(x)
        #print('after fully connected 2:          ', x.size())
        
        x = torch.sigmoid(x)
        #print('after sigmoid:                    ', x.size())
        
        #print('\n##########################################################\n')
        #print('x[0].size():                      ', x[0].size())
        #print('x.size():                         ', x.size())
        return x