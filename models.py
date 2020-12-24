import torch
import torch.nn as nn
import torch.nn.functional as F
from config import audio_channels, working_sample_rate, mel_bins, hop_size, classes_num

DEFAULT_CHANNEL_AND_POOL=[(64,2), (128,2), (256,2), (512,1)]

def interpolate(x, ratio):
    '''
    Upscales the 2nd axis of x by 'ratio', i.e Repeats each element in it 'ratio' times:
    In other words: Interpolate the prediction to have the same time_steps as the target.
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=2):
        super(ConvBlock, self).__init__()
        self.pool_size = pool_size
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

    def forward(self, input):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        x = F.avg_pool2d(x, kernel_size=self.pool_size)

        return x


class Cnn_AvgPooling(nn.Module):
    def __init__(self, classes_num, model_config=DEFAULT_CHANNEL_AND_POOL):
        super(Cnn_AvgPooling, self).__init__()
        self.model_config = model_config
        self.num_pools = 1 if model_config[0][1] == 2 else 1
        self.conv_blocks = [ConvBlock(in_channels=audio_channels, out_channels=model_config[0][0], pool_size=model_config[0][1])]
        for i in range(1, len(model_config)):
            pool_size = model_config[i][1]
            if pool_size == 2:
                self.num_pools += 1
            self.conv_blocks.append(ConvBlock(in_channels=model_config[i - 1][0], out_channels=model_config[i][0], pool_size=pool_size))

        self.conv_blocks = torch.nn.Sequential(*self.conv_blocks)

        self.event_fc = nn.Linear(model_config[-1][0], classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.event_fc)

    def forward(self, input):
        '''
        Input: (channels_num, batch_size, times_steps, freq_bins)'''

        x = input.transpose(0, 1)  # -> (batch_size, channels_num, times_steps, freq_bins)

        x = self.conv_blocks(x)

        # x.shape : (batch_size, channels_num, times_steps, freq_bins)

        x = torch.mean(x, dim=3)    # (batch_size, channels_num, time_steps)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels_num)

        event_output = torch.sigmoid(self.event_fc(x))  # (batch_size, time_steps, classes_num)

        # Interpolate
        event_output = interpolate(event_output, 2**(self.num_pools))

        return event_output

    def model_description(self):
        print("Model description")
        b = 'b'
        w = mel_bins
        h = 60 * working_sample_rate // hop_size
        c = audio_channels
        # dummy_input = torch.ones()
        print(f"\tInput: ({b}, {c}, {h}, {w})")
        for (c, k) in self.model_config:
            h = h // k
            w = w // k
            print(f"\tconv_block -> ({b}, {c}, {h}, {w})")

        print(f"\tmean(dim=3) -> ({b}, {c}, {h})")
        print(f"\ttranspose(1,2) -> ({b}, {h}, {c})")
        print(f"\tFC + sigmoid -> ({b}, {h}, {classes_num})")
        h *= 2**(self.num_pools)
        print(f"\tinterpolate({2**(self.num_pools)})-> ({b}, {h}, {classes_num})")
