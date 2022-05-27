import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants.constants as constants
import contextlib
from torchsummary import summary
#pip install torchsummary

def write_model_1(file_path, model, D):
        with open(file_path, "a") as o:
            with contextlib.redirect_stdout(o):
                summary(model, (constants.prosody_size, 300), batch_size = constants.batch_size)
        o.close()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(constants.prosody_size, 64, constants.first_kernel_size, padding = constants.first_padding_size, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, constants.kernel_size, padding = constants.padding_size, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128 , 256 , constants.kernel_size,  padding = constants.padding_size, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, constants.kernel_size, padding = constants.padding_size, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128, 64, constants.kernel_size, padding = constants.padding_size, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, constants.pose_size + constants.au_size, constants.kernel_size, padding = constants.padding_size, bias=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x