import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constant
import contextlib
from torchsummary import summary
from utils.model_parts import DoubleConv, Down, OutConv, Up
#pip install torchsummary

def write_model_3(file_path, model, D):
        with open(file_path, "a") as o:
            with contextlib.redirect_stdout(o):
                summary(model, (constant.prosody_size, 300), batch_size = constant.batch_size)
        o.close()


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__() 
        bilinear = True
        factor = 2 if bilinear else 1
        ##Encoder

        self.inc = DoubleConv(constant.prosody_size, 64, constant.first_kernel_size, constant.first_padding_size)
        self.down1 = Down(64, 128, constant.kernel_size, constant.padding_size)
        self.down2 = Down(128, 256, constant.kernel_size, constant.padding_size)
        self.down3 = Down(256, 512, constant.kernel_size, constant.padding_size)
        self.down4 = Down(512, 1024 // factor, constant.kernel_size, constant.padding_size)

        
        ##Decoder pose
        self.up1_pose = Up(1024, 512 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up2_pose = Up(512, 256 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up3_pose = Up(256, 128 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up4_pose = Up(128, 64, constant.kernel_size, constant.padding_size, bilinear)
        self.outc_pose = OutConv(64, constant.pose_size, constant.kernel_size, constant.padding_size)

        ##Decoder AUs
        self.up1_au = Up(1024, 512 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up2_au = Up(512, 256 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up3_au = Up(256, 128 // factor, constant.kernel_size, constant.padding_size, bilinear)
        self.up4_au = Up(128, 64, constant.kernel_size, constant.padding_size, bilinear)
        self.outc_au = OutConv(64, constant.au_size, constant.kernel_size, constant.padding_size)


    def forward(self, x):
        #Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        #Decoder pose and gaze angle
        x = self.up1_pose(x5, x4)
        x = self.up2_pose(x, x3)
        x = self.up3_pose(x, x2)
        x = self.up4_pose(x, x1)
        logits_pose = self.outc_pose(x)
        logits_pose = torch.sigmoid(logits_pose)

        #Decoder AUs
        x = self.up1_au(x5, x4)
        x = self.up2_au(x, x3)
        x = self.up3_au(x, x2)
        x = self.up4_au(x, x1)
        logits_au = self.outc_au(x)
        logits_au = torch.sigmoid(logits_au)
        
        return logits_pose, logits_au