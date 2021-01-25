from torch.nn import Sequential
from dataset.waveform.waveform_configs import frame_size, audio_channels
import torch.nn as nn
import torch

from utils.common import count_parameters, human_format


class M3(nn.Module):
    def __init__(self, classes_num):
        super(M3, self).__init__()
        self.conv_block1 = Sequential(
            nn.Conv1d(audio_channels, 64, kernel_size=79, stride=4, padding=39),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4,4)
        )

        self.conv_block2 = Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, 4)
        )
        self.conv_block3 = Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4, 4)
        )
        self.conv_block4 = Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4, 4)
        )
        self.conv_block5 = Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        # x: (b, c, frame_size)
        x = self.conv_block1(x)  # x: (b, 64, frame_size / 16)
        x = self.conv_block2(x)  # x: (b, 64, frame_size / 64)
        x = self.conv_block3(x)  # x: (b, 64, frame_size / 256)
        x = self.conv_block4(x)  # x: (b, 64, frame_size / 1024)
        x = self.conv_block5(x)  # x: (b, 64, frame_size / 1024)
        x = torch.mean(x, dim=2)
        x = self.fc(x)

        # x = torch.sigmoid(x)

        return x

    def model_description(self):
        print("Waveform model:")
        print(f"\t- Model has {human_format(count_parameters(self))} parameters")