import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constant


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(constant.prosody_size, 64, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128 , 256 , 3,  stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128, 64, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, constant.pose_size + constant.au_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x