import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants.constants as constants
import contextlib
from torchsummary import summary
from utils.model_parts import DoubleConv, Down, OutConv, Up
#pip install torchsummary

def write_model_12(file_path, model, D):
        with open(file_path, "a") as o:
            with contextlib.redirect_stdout(o):
                summary(model, (constants.prosody_size, 300), batch_size = constants.batch_size)
        o.close()


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__() 
        bilinear = True
        factor = 2 if bilinear else 1

        ##Encoder
        self.inc = DoubleConv(constants.prosody_size, 64, constants.first_kernel_size, constants.first_padding_size)
        self.down1 = Down(64, 128, constants.kernel_size, constants.padding_size)
        self.down2 = Down(128, 256, constants.kernel_size, constants.padding_size)
        self.down3 = Down(256, 512, constants.kernel_size, constants.padding_size)
        self.down4 = Down(512, 1024 // factor, constants.kernel_size, constants.padding_size)

        ##Decoder pose_au
        self.up1_pose_au = Up(1024, 512 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up2_pose_au = Up(512, 256 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up3_pose_au = Up(256, 128 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up4_pose_au = Up(128, 64, constants.kernel_size, constants.padding_size, bilinear)
        self.outc_au = OutConv(64, constants.au_size, constants.kernel_size, constants.padding_size)


    def forward(self, x):
        #Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        #Decoder pose_au
        x = self.up1_pose_au(x5, x4)
        x = self.up2_pose_au(x, x3)
        x = self.up3_pose_au(x, x2)
        x = self.up4_pose_au(x, x1)
        logits_pose_au = self.outc_au(x)
        logits_pose_au = torch.sigmoid(logits_pose_au)

        return logits_pose_au
