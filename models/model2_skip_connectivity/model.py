import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constant
import contextlib
from torchsummary import summary
from utils.model_parts import DoubleConv, Down, OutConv, Up
#pip install torchsummary

def write_model_2(file_path, model, D):
        with open(file_path, "a") as o:
            with contextlib.redirect_stdout(o):
                summary(model, (constant.prosody_size, 300), batch_size = constant.batch_size)
        o.close()


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        bilinear = True

        self.inc = DoubleConv(constant.prosody_size, 64, constant.first_kernel_size, constant.first_padding_size)
        self.down1 = Down(64, 128, constant.kernel_size, constant.padding_size)
        self.down2 = Down(128, 256, constant.kernel_size, constant.padding_size)
        self.down3 = Down(256, 512, constant.kernel_size, constant.padding_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, constant.kernel_size, constant.padding_size)
        self.up1 = Up(1024, 512 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up2 = Up(512, 256 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up3 = Up(256, 128 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up4 = Up(128, 64, constant.kernel_size, constant.padding_size, bilinear)
        self.outc = OutConv(64, constant.pose_size + constant.au_size, constant.kernel_size, constant.padding_size)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.sigmoid(logits)
        return logits