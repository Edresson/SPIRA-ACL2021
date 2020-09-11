<<<<<<< HEAD
# based on 
# https://github.com/rogertrullo/pytorch_convlstm/blob/master/conv_lstm.py 
# and 
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
# (* needs revision) -> something that I annotated and can be wrong

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.generic_utils import Mish

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.data.fill_(0)

        
        
class utfpr_convLSTM_cell_v1(nn.Module):
    """
        **Initialize a basic convolutional LSTM cell**

        -ARGS:
            shape: int tuple containing the hidden states h and c width and height values
            input_channels: int containing the number of channels in the input (* needs revision)
            filter_size: int containing the convolution filters height and width
            num_features: int containing the number of channels in each state, like hidden_size

    """
    def __init__(self, shape, input_channels, num_filters, filter_size, num_features):
        super(utfpr_convLSTM_cell_v1, self).__init__()
        self.shape          = (shape)               # (hidden_states_h_and_c_height, hidden_states_h_and_c_width)
        self.input_channels = input_channels        # in case of RGB images, we have 3 (red, green and blue, respectively). For MFCCs, we have 1 input channel, which is [seq_len, num_features]
        self.num_filters    = num_filters
        self.filter_size    = filter_size           # actually kernel size
        self.num_filters    = num_filters 
        self.num_features   = num_features          # means the height (nem number)
        self.padding        = int((filter_size - 1) / 2) # guarantees output dim = input dim
        self.conv           = nn.Conv2d(self.input_channels + self.num_filters, 4 * self.num_filters, self.filter_size, 1, self.padding) # 2 input channels: input and hidden state
        
        # TODO: experiment with some of Edresson's spiraconv convolutions
        """convs = [
            # cnn1
            nn.Conv2d(self.input_channels + self.num_filters, 4 * self.num_filters, self.filter_size, dilation=(2, 1)),
            nn.GroupNorm(16, 32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            
            # cnn2
            nn.Conv2d(4 * self.num_filters, 2 * self.num_filters, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 

            # cnn3
            nn.Conv2d(2 * self.num_filters, self.num_filters, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.GroupNorm(4, 8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7)]

        self.conv = nn.Sequential(*convs)"""

        
        
        
    def forward(self, input, hidden_s): # input is not affected by unsqueeze in spiraconv.py line 75, for some reason
        # input_need: (batch_size        , num_channels, sequence_step_size, num_features)
        # input_have: (sequence_step_size, num_channels, batch_size        , num_features)
        
        #print('\n############################ Inside utfpr_convLSTM_cell #############################')
        #print('input: ', input.size()) 
        input = input.transpose(0, 2)   # transpose to use current_input[t,...] in line 171
        input = input.cuda() 
        
        hidden_state, hidden_state_c = hidden_s
        hidden_state.cuda()
        hidden_state_c.cuda()
        
        #print('hidden_state size:   ', hidden_state.size())
        #print('hidden_state_c size: ', hidden_state_c.size())
        #print('input size:          ', input.size())
        
        #print('\n')
        
        #print('self.shape: '    , self.shape, '\n')
        #print('input_channels: ', self.input_channels, '\n')
        #print('filter_size: '   , self.filter_size, '\n')
        #print('num_features: '  , self.num_features, '\n')
        #print('padding: '       , self.padding, '\n')
        
        concatenated                 = torch.cat((input, hidden_state), 1) # concatenate
        #print('concatenated.size(): ', concatenated.size())
        #print('self.input_channels: ', self.input_channels)
        #print('self.num_filters:    ', self.num_filters)
        concatenated_post_conv       = self.conv(concatenated)
        #print("ConcatAftConv.size():", concatenated_post_conv.size())
        (ai, af, ao, ag)             = torch.split(concatenated_post_conv, self.num_filters, dim=1)
        #print("ai.size()", ai.size())
        #print("af.size()", af.size())
        #print("ao.size()", ao.size())
        #print("ag.size()", ag.size())
        i                            = torch.sigmoid(ai)
        f                            = torch.sigmoid(af)
        o                            = torch.sigmoid(ao)
        g                            = torch.tanh(ag)
        
        next_hidden_state_c          = f * hidden_state_c + i * g
        next_hidden_state            = o * torch.tanh(next_hidden_state_c)  
        return next_hidden_state, next_hidden_state_c
        
        
        
    def init_hidden_states(self, batch_size):
        # input: [batch_size, number_channels, seq_len, num_features]
        
        # number_channels is from images. As we are analyzing an MFCC, we have a single channel, which is a [seq_len, num_features] matrix. This dimension turns into num_filters when a convolution is applied to it.
        #                            batch_size, num_feature_maps , seq_length   , num_features
        #return (Variable(torch.zeros(batch_size, self.num_filters, self.shape[2], self.shape[3])).cuda(), Variable(torch.zeros(batch_size, self.num_filters , self.shape[2], self.shape[3])).cuda()) 
        return (Variable(torch.rand(batch_size, self.num_filters, self.shape[2], self.shape[3])).cuda(), Variable(torch.rand(batch_size, self.num_filters , self.shape[2], self.shape[3])).cuda()) 
    # TODO: initialize with torch.rand to avoid training problems, as described in https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """Using a zero-valued initial state can also result in overfitting, though in a different way. Ordinarily, losses at the early steps of a sequence-to-sequence model (i.e., those immediately after a state reset) will be larger than those at later steps, because there is less history. Thus, their contribution to the gradient during learning will be relatively higher. But if all state resets are associated with a zero-state, the model can (and will) learn how to compensate for precisely this. As the ratio of state resets to total observations increases, the model parameters will become increasingly tuned to this zero state, which may affect performance on later time steps.

One simple solution is to make the initial state noisy. This is the approach suggested by Zimmerman et al. (2012), who take it even a step further by making the magnitude of the initial state noise change according to the backpropagated error."""
    
    
    
class utfpr_convLSTM_v1(nn.Module):
    """
    **Initialize a basic convolutional LSTM call**
    
    -ARGS:
        shape: int tuple containing the hidden states h and c width and height values
        input_channels: int containing the number of channels in the input (* needs revision)
        filter_size: int containing the convolution filters height and width
        num_features: int containing the number of channels in each state, like hidden_size
        num_layers: int containing the number of convolutional LSTM layers/cells (* needs revision)
    """   
       
    def __init__(self, shape, input_channels, num_filters, filter_size, num_features, num_layers, sequence_step_size):
        super(utfpr_convLSTM_v1, self).__init__()
        self.shape              = shape          # (hidden_states_h_and_c_height, hidden_states_h_and_c_width)
        self.input_channels     = input_channels # in case of images, which have multiple channels
        self.num_filters        = num_filters
        self.filter_size        = filter_size
        self.num_features       = num_features
        self.num_layers         = num_layers
        self.sequence_step_size = sequence_step_size   # MFCC x axis interval to be analyzed per time step in the convLSTM
        
        cell_list               = []
        cell_list.append(utfpr_convLSTM_cell_v1(self.shape, 
                                                self.input_channels, 
                                                self.num_filters, #current number of feature maps (as none convolutional layers were used yet, we only have the single MFCC channel, which is a [seq_len, num_features] tensor)
                                                self.filter_size, 
                                                self.num_features)
                 .cuda())            # first convLSTM cell has a different number of input channels
        for idcell in range(1, self.num_layers):
            cell_list.append(utfpr_convLSTM_cell_v1(self.shape, 
                                                    self.num_filters,
                                                    self.num_filters,
                                                    self.filter_size, 
                                                    self.num_features)
                 .cuda())
        
        self.cell_list          = nn.ModuleList(cell_list)
        
        
        
    def forward(self, input, hidden_state):
        """
        -ARGS:
            input: tensor with shape (batch_size, sequence_length (timestamp, number_channels, hidden_states_h_and_c_height, hidden_states_h_and_c_width) (* needs revision - swapped batch_size and sequence_len, as it was similiar to the current_input in line 98, after input.transpose(0, 1))
            hidden_state: list of tuples, one for every layer, with shape (hidden_state_layer_i, hidden_state_c_layer_i)
        """
        # input_need: (MFCC_width, batch_size, input_channels/num_features, MFCC_height) # MFCC_width = timestamp[t:t+sequence_step_size] and MFCC_height = num_features #                                                          #
        # input_have: (batch_size, 1, timestamp, num_features)                           # input_have[1] = num_channels, which becomes num_filters after convolution     # batch_size and num_features are specified in config.json #
        
        #print('\n############################ Inside utfpr_convLSTM #############################')
        #print('self.shape:   ', self.shape)
        #print('current_input init   ', input.size())
        
        current_input     = input.transpose(0, 2) # (1, batch_size, timestamp, num_features) # here, we have 1 channel (position [0]), as it is a single [seq_len, num_features] matrix (MFCC) # transpose to use current_input[t,...] in line 171
        #print('current_input transp1', current_input.size())
        
        #current_input      = current_input.transpose(1, 2) # ()
        #print('current_input transp1', current_input.size())
        
        next_hidden_state = [] # next hidden states h and c
        sequence_length   = current_input.size(0)
        #print(sequence_length)
        
        for idlayer in range(self.num_layers): # loop for every layer
            hidden_state_c       = hidden_state[idlayer]
            all_output           = []
            output_inner         = []
            
            # slice MFCC into 500 ms intervals (padding when data isn't 500 ms, usually at the end of inputs)
            for t in range(0, sequence_length, self.sequence_step_size): # loop for each audio segment with shape [self.sequence_step_size, self.num_features]
                if (t + self.sequence_step_size <= sequence_length):
                    #print('\n############################ Inside utfpr_convLSTM #############################')
                    # no padding needed
                    #print('\nsliced input (t): ', current_input[t:t+self.sequence_step_size,...].size())
                    hidden_state_c = self.cell_list[idlayer](current_input[t:t + self.sequence_step_size, ...], hidden_state_c) # cell list have conv_lstm_cells for every layer
                    
                else:
                    #print('\n############################ Inside utfpr_convLSTM #############################')
                    # padding needed from 
                    #print(sequence_length - t) # current_input length
                    padding_length                               = t + self.sequence_step_size - sequence_length        
                    zero_padded_input                            = torch.zeros((sequence_length - t + padding_length, current_input.size(1), current_input.size(2), current_input.size(3)))
                    #print('\nzero_padded_input.size():                  ', zero_padded_input.size())
                    #print('curr_inp[t:t+self.sequence_step_size].size():', current_input[t:t+self.sequence_step_size].size())
                    
                    zero_padded_input[:sequence_length - t, ...] = current_input[t:, ...] 
                    #current_input[sequence_length:sequence_length+padding_length, ...] = zero_padded_input[sequence_length-t:, ...] # RuntimeError: The expanded size of the tensor (0) must match the existing size (206) at non-singleton dimension 0.  Target sizes: [0, 1, 1, 40].  Tensor sizes: [206, 1, 1, 40]
                    #print('\nzero_padded_input length:                  ', zero_padded_input.size())
                    #print('curr_inp[t:t+self.sequence_step_size].size():', current_input[t:t+self.sequence_step_size].size())
                    
                    #print('\nsliced input length (t):                   ', zero_padded_input.size())
                    #print('\nsliced padded input (t):                   ', zero_padded_input.size())
                    hidden_state_c                               = self.cell_list[idlayer](zero_padded_input, hidden_state_c) # cell list have conv_lstm_cells for every layer
                
                #print('\nhidden_state_c[0].size(): ', hidden_state_c[0].size())
                output_inner.append(hidden_state_c[0])
                #print(len(output_inner))
            
            next_hidden_state.append(hidden_state_c)
            
            #print('current_input.size():  '        , current_input.size())
            #print('output_inner[0].size():'        , output_inner[0].size())
            #print('output_inner[1].size():'        , output_inner[1].size())
            #print('len(output_inner):     '        , len(output_inner))
            #print(*output_inner[0].size()) # just as output_inner[0].size, but instead of printing torch.Size([1, 40, 500, 40]), it prints simply 1 40 500 40
            
            padded_hidden_out    = torch.cat(output_inner, 2)
            #print('padded_hidden_out.size():      ', padded_hidden_out.size())
            
            unpadded_hidden_out  = padded_hidden_out[:, :, :current_input.size(0), :] 
            #print('unpadded_hidden_out.size():    ', unpadded_hidden_out.size())
            
            unpadded_hidden_out  = unpadded_hidden_out.transpose(0, 2)
            #print('unpadded_hidden_out.size():    ', unpadded_hidden_out.size())
            
            # print('unpadded_hidden_out:', unpadded_hidden_out)
            
            #print('unpadded_hidden_out[0].size(): ', unpadded_hidden_out.size())
            #print('*unpadded_hidden_out[0].size():', *unpadded_hidden_out.size())
            current_input  = unpadded_hidden_out.view(current_input.size(0), *unpadded_hidden_out[0].size()) # (sequence_length, batch_size, number_channels, hidden_states_h_and_c_height, hidden_states_h_and_c_width) # concat 794 (output_inner) in pos 2 (with seq_len)
            #print('current_input.size():          ', current_input.size())
        
        return next_hidden_state, current_input
    
    
    
    def init_hidden_states(self, batch_size):
        init_states = [] # list fo tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden_states(batch_size))
=======
# based on 
# https://github.com/rogertrullo/pytorch_convlstm/blob/master/conv_lstm.py 
# and 
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
# (* needs revision) -> something that I annotated and can be wrong

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.generic_utils import Mish

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.data.fill_(0)

        
        
class utfpr_convLSTM_cell_v1(nn.Module):
    """
        **Initialize a basic convolutional LSTM cell**

        -ARGS:
            shape: int tuple containing the hidden states h and c width and height values
            input_channels: int containing the number of channels in the input (* needs revision)
            filter_size: int containing the convolution filters height and width
            num_features: int containing the number of channels in each state, like hidden_size

    """
    def __init__(self, shape, input_channels, num_filters, filter_size, num_features):
        super(utfpr_convLSTM_cell_v1, self).__init__()
        self.shape          = (shape)               # (hidden_states_h_and_c_height, hidden_states_h_and_c_width)
        self.input_channels = input_channels        # in case of RGB images, we have 3 (red, green and blue, respectively). For MFCCs, we have 1 input channel, which is [seq_len, num_features]
        self.num_filters    = num_filters
        self.filter_size    = filter_size           # actually kernel size
        self.num_filters    = num_filters 
        self.num_features   = num_features          # means the height (nem number)
        self.padding        = int((filter_size - 1) / 2) # guarantees output dim = input dim
        self.conv           = nn.Conv2d(self.input_channels + self.num_filters, 4 * self.num_filters, self.filter_size, 1, self.padding) # 2 input channels: input and hidden state
        
        # TODO: experiment with some of Edresson's spiraconv convolutions
        """convs = [
            # cnn1
            nn.Conv2d(self.input_channels + self.num_filters, 4 * self.num_filters, self.filter_size, dilation=(2, 1)),
            nn.GroupNorm(16, 32), Mish(), nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 
            
            # cnn2
            nn.Conv2d(4 * self.num_filters, 2 * self.num_filters, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7), 

            # cnn3
            nn.Conv2d(2 * self.num_filters, self.num_filters, kernel_size=(3, 1), dilation=(2, 1)), 
            nn.GroupNorm(4, 8), Mish(),  nn.MaxPool2d(kernel_size=(2,1)), nn.Dropout(p=0.7)]

        self.conv = nn.Sequential(*convs)"""

        
        
        
    def forward(self, input, hidden_s): # input is not affected by unsqueeze in spiraconv.py line 75, for some reason
        # input_need: (batch_size        , num_channels, sequence_step_size, num_features)
        # input_have: (sequence_step_size, num_channels, batch_size        , num_features)
        
        #print('\n############################ Inside utfpr_convLSTM_cell #############################')
        #print('input: ', input.size()) 
        input = input.transpose(0, 2)   # transpose to use current_input[t,...] in line 171
        input = input.cuda() 
        
        hidden_state, hidden_state_c = hidden_s
        hidden_state.cuda()
        hidden_state_c.cuda()
        
        #print('hidden_state size:   ', hidden_state.size())
        #print('hidden_state_c size: ', hidden_state_c.size())
        #print('input size:          ', input.size())
        
        #print('\n')
        
        #print('self.shape: '    , self.shape, '\n')
        #print('input_channels: ', self.input_channels, '\n')
        #print('filter_size: '   , self.filter_size, '\n')
        #print('num_features: '  , self.num_features, '\n')
        #print('padding: '       , self.padding, '\n')
        
        concatenated                 = torch.cat((input, hidden_state), 1) # concatenate
        #print('concatenated.size(): ', concatenated.size())
        #print('self.input_channels: ', self.input_channels)
        #print('self.num_filters:    ', self.num_filters)
        concatenated_post_conv       = self.conv(concatenated)
        #print("ConcatAftConv.size():", concatenated_post_conv.size())
        (ai, af, ao, ag)             = torch.split(concatenated_post_conv, self.num_filters, dim=1)
        #print("ai.size()", ai.size())
        #print("af.size()", af.size())
        #print("ao.size()", ao.size())
        #print("ag.size()", ag.size())
        i                            = torch.sigmoid(ai)
        f                            = torch.sigmoid(af)
        o                            = torch.sigmoid(ao)
        g                            = torch.tanh(ag)
        
        next_hidden_state_c          = f * hidden_state_c + i * g
        next_hidden_state            = o * torch.tanh(next_hidden_state_c)  
        return next_hidden_state, next_hidden_state_c
        
        
        
    def init_hidden_states(self, batch_size):
        # input: [batch_size, number_channels, seq_len, num_features]
        
        # number_channels is from images. As we are analyzing an MFCC, we have a single channel, which is a [seq_len, num_features] matrix. This dimension turns into num_filters when a convolution is applied to it.
        #                            batch_size, num_feature_maps , seq_length   , num_features
        #return (Variable(torch.zeros(batch_size, self.num_filters, self.shape[2], self.shape[3])).cuda(), Variable(torch.zeros(batch_size, self.num_filters , self.shape[2], self.shape[3])).cuda()) 
        return (Variable(torch.rand(batch_size, self.num_filters, self.shape[2], self.shape[3])).cuda(), Variable(torch.rand(batch_size, self.num_filters , self.shape[2], self.shape[3])).cuda()) 
    # TODO: initialize with torch.rand to avoid training problems, as described in https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """Using a zero-valued initial state can also result in overfitting, though in a different way. Ordinarily, losses at the early steps of a sequence-to-sequence model (i.e., those immediately after a state reset) will be larger than those at later steps, because there is less history. Thus, their contribution to the gradient during learning will be relatively higher. But if all state resets are associated with a zero-state, the model can (and will) learn how to compensate for precisely this. As the ratio of state resets to total observations increases, the model parameters will become increasingly tuned to this zero state, which may affect performance on later time steps.

One simple solution is to make the initial state noisy. This is the approach suggested by Zimmerman et al. (2012), who take it even a step further by making the magnitude of the initial state noise change according to the backpropagated error."""
    
    
    
class utfpr_convLSTM_v1(nn.Module):
    """
    **Initialize a basic convolutional LSTM call**
    
    -ARGS:
        shape: int tuple containing the hidden states h and c width and height values
        input_channels: int containing the number of channels in the input (* needs revision)
        filter_size: int containing the convolution filters height and width
        num_features: int containing the number of channels in each state, like hidden_size
        num_layers: int containing the number of convolutional LSTM layers/cells (* needs revision)
    """   
       
    def __init__(self, shape, input_channels, num_filters, filter_size, num_features, num_layers, sequence_step_size):
        super(utfpr_convLSTM_v1, self).__init__()
        self.shape              = shape          # (hidden_states_h_and_c_height, hidden_states_h_and_c_width)
        self.input_channels     = input_channels # in case of images, which have multiple channels
        self.num_filters        = num_filters
        self.filter_size        = filter_size
        self.num_features       = num_features
        self.num_layers         = num_layers
        self.sequence_step_size = sequence_step_size   # MFCC x axis interval to be analyzed per time step in the convLSTM
        
        cell_list               = []
        cell_list.append(utfpr_convLSTM_cell_v1(self.shape, 
                                                self.input_channels, 
                                                self.num_filters, #current number of feature maps (as none convolutional layers were used yet, we only have the single MFCC channel, which is a [seq_len, num_features] tensor)
                                                self.filter_size, 
                                                self.num_features)
                 .cuda())            # first convLSTM cell has a different number of input channels
        for idcell in range(1, self.num_layers):
            cell_list.append(utfpr_convLSTM_cell_v1(self.shape, 
                                                    self.num_filters,
                                                    self.num_filters,
                                                    self.filter_size, 
                                                    self.num_features)
                 .cuda())
        
        self.cell_list          = nn.ModuleList(cell_list)
        
        
        
    def forward(self, input, hidden_state):
        """
        -ARGS:
            input: tensor with shape (batch_size, sequence_length (timestamp, number_channels, hidden_states_h_and_c_height, hidden_states_h_and_c_width) (* needs revision - swapped batch_size and sequence_len, as it was similiar to the current_input in line 98, after input.transpose(0, 1))
            hidden_state: list of tuples, one for every layer, with shape (hidden_state_layer_i, hidden_state_c_layer_i)
        """
        # input_need: (MFCC_width, batch_size, input_channels/num_features, MFCC_height) # MFCC_width = timestamp[t:t+sequence_step_size] and MFCC_height = num_features #                                                          #
        # input_have: (batch_size, 1, timestamp, num_features)                           # input_have[1] = num_channels, which becomes num_filters after convolution     # batch_size and num_features are specified in config.json #
        
        #print('\n############################ Inside utfpr_convLSTM #############################')
        #print('self.shape:   ', self.shape)
        #print('current_input init   ', input.size())
        
        current_input     = input.transpose(0, 2) # (1, batch_size, timestamp, num_features) # here, we have 1 channel (position [0]), as it is a single [seq_len, num_features] matrix (MFCC) # transpose to use current_input[t,...] in line 171
        #print('current_input transp1', current_input.size())
        
        #current_input      = current_input.transpose(1, 2) # ()
        #print('current_input transp1', current_input.size())
        
        next_hidden_state = [] # next hidden states h and c
        sequence_length   = current_input.size(0)
        #print(sequence_length)
        
        for idlayer in range(self.num_layers): # loop for every layer
            hidden_state_c       = hidden_state[idlayer]
            all_output           = []
            output_inner         = []
            
            # slice MFCC into 500 ms intervals (padding when data isn't 500 ms, usually at the end of inputs)
            for t in range(0, sequence_length, self.sequence_step_size): # loop for each audio segment with shape [self.sequence_step_size, self.num_features]
                if (t + self.sequence_step_size <= sequence_length):
                    #print('\n############################ Inside utfpr_convLSTM #############################')
                    # no padding needed
                    #print('\nsliced input (t): ', current_input[t:t+self.sequence_step_size,...].size())
                    hidden_state_c = self.cell_list[idlayer](current_input[t:t + self.sequence_step_size, ...], hidden_state_c) # cell list have conv_lstm_cells for every layer
                    
                else:
                    #print('\n############################ Inside utfpr_convLSTM #############################')
                    # padding needed from 
                    #print(sequence_length - t) # current_input length
                    padding_length                               = t + self.sequence_step_size - sequence_length        
                    zero_padded_input                            = torch.zeros((sequence_length - t + padding_length, current_input.size(1), current_input.size(2), current_input.size(3)))
                    #print('\nzero_padded_input.size():                  ', zero_padded_input.size())
                    #print('curr_inp[t:t+self.sequence_step_size].size():', current_input[t:t+self.sequence_step_size].size())
                    
                    zero_padded_input[:sequence_length - t, ...] = current_input[t:, ...] 
                    #current_input[sequence_length:sequence_length+padding_length, ...] = zero_padded_input[sequence_length-t:, ...] # RuntimeError: The expanded size of the tensor (0) must match the existing size (206) at non-singleton dimension 0.  Target sizes: [0, 1, 1, 40].  Tensor sizes: [206, 1, 1, 40]
                    #print('\nzero_padded_input length:                  ', zero_padded_input.size())
                    #print('curr_inp[t:t+self.sequence_step_size].size():', current_input[t:t+self.sequence_step_size].size())
                    
                    #print('\nsliced input length (t):                   ', zero_padded_input.size())
                    #print('\nsliced padded input (t):                   ', zero_padded_input.size())
                    hidden_state_c                               = self.cell_list[idlayer](zero_padded_input, hidden_state_c) # cell list have conv_lstm_cells for every layer
                
                #print('\nhidden_state_c[0].size(): ', hidden_state_c[0].size())
                output_inner.append(hidden_state_c[0])
                #print(len(output_inner))
            
            next_hidden_state.append(hidden_state_c)
            
            #print('current_input.size():  '        , current_input.size())
            #print('output_inner[0].size():'        , output_inner[0].size())
            #print('output_inner[1].size():'        , output_inner[1].size())
            #print('len(output_inner):     '        , len(output_inner))
            #print(*output_inner[0].size()) # just as output_inner[0].size, but instead of printing torch.Size([1, 40, 500, 40]), it prints simply 1 40 500 40
            
            padded_hidden_out    = torch.cat(output_inner, 2)
            #print('padded_hidden_out.size():      ', padded_hidden_out.size())
            
            unpadded_hidden_out  = padded_hidden_out[:, :, :current_input.size(0), :] 
            #print('unpadded_hidden_out.size():    ', unpadded_hidden_out.size())
            
            unpadded_hidden_out  = unpadded_hidden_out.transpose(0, 2)
            #print('unpadded_hidden_out.size():    ', unpadded_hidden_out.size())
            
            # print('unpadded_hidden_out:', unpadded_hidden_out)
            
            #print('unpadded_hidden_out[0].size(): ', unpadded_hidden_out.size())
            #print('*unpadded_hidden_out[0].size():', *unpadded_hidden_out.size())
            current_input  = unpadded_hidden_out.view(current_input.size(0), *unpadded_hidden_out[0].size()) # (sequence_length, batch_size, number_channels, hidden_states_h_and_c_height, hidden_states_h_and_c_width) # concat 794 (output_inner) in pos 2 (with seq_len)
            #print('current_input.size():          ', current_input.size())
        
        return next_hidden_state, current_input
    
    
    
    def init_hidden_states(self, batch_size):
        init_states = [] # list fo tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden_states(batch_size))
>>>>>>> fb30d9d359100652a6885b49325fad633c43ccdc
        return init_states