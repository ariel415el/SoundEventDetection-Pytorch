import torch
import torch.nn as nn
import torch.nn.functional as F
from config import audio_channels

def interpolate(x, ratio):
    '''Interpolate the prediction to have the same time_steps as the target.
    The time_steps mismatch is caused by maxpooling in CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to upsample
    '''
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class Cnn_2layers_AvgPooling(nn.Module):

    def __init__(self, classes_num):
        super(Cnn_2layers_AvgPooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=audio_channels, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)


        self.event_fc = nn.Linear(128, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.event_fc)

        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        '''
        Input: (channels_num, batch_size, times_steps, freq_bins)'''

        interpolate_ratio = 32

        x = input.transpose(0, 1)
        '''(batch_size, channels_num, times_steps, freq_bins)'''

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_steps)
        x = x.transpose(1, 2)  # (batch_size, time_steps, feature_maps)

        event_output = torch.sigmoid(self.event_fc(x))  # (batch_size, time_steps, classes_num)

        # Interpolate
        event_output = interpolate(event_output, interpolate_ratio)

        return event_output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
    
class Cnn_9layers_AvgPooling(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=audio_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.event_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.event_fc)

    def forward(self, input):
        '''
        Input: (channels_num, batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input.transpose(0, 1)
        '''(batch_size, channels_num, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)    # (batch_size, feature_maps, time_steps)
        x = x.transpose(1, 2)   # (batch_size, time_steps, feature_maps)
        
        event_output = torch.sigmoid(self.event_fc(x))  # (batch_size, time_steps, classes_num)

        # Interpolate
        event_output = interpolate(event_output, interpolate_ratio)


        return event_output
