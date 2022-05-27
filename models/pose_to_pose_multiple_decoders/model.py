import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants.constants as constants
import contextlib
from torchsummary import summary
from utils.model_parts import DoubleConv, Down, OutConv, Up
#pip install torchsummary

def write_model_8(file_path, model, D):
        with open(file_path, "a") as o:
            with contextlib.redirect_stdout(o):
                summary(model, (constants.pose_size + constants.au_size, 300), batch_size = constants.batch_size)
        o.close() 

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        bilinear = True
        factor = 2 if bilinear else 1

        ##Encoder
        self.inc = DoubleConv(constants.pose_size + constants.au_size, 64, constants.first_kernel_size, constants.first_padding_size)
        self.down1 = Down(64, 128, constants.kernel_size, constants.padding_size)
        self.down2 = Down(128, 256, constants.kernel_size, constants.padding_size)
        self.down3 = Down(256, 512, constants.kernel_size, constants.padding_size)
        self.down4 = Down(512, 1024 // factor, constants.kernel_size, constants.padding_size)

        
        ##Decoder eye
        self.up1_eye = Up(1024, 512 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up2_eye = Up(512, 256 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up3_eye = Up(256, 128 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up4_eye = Up(128, 64, constants.kernel_size, constants.padding_size, bilinear)
        self.outc_eye = OutConv(64, constants.eye_size, constants.kernel_size, constants.padding_size)


        ##Decoder pose_t
        self.up1_pose_t = Up(1024, 512 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up2_pose_t = Up(512, 256 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up3_pose_t = Up(256, 128 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up4_pose_t = Up(128, 64, constants.kernel_size, constants.padding_size, bilinear)
        self.outc_pose_t = OutConv(64, constants.pose_t_size, constants.kernel_size, constants.padding_size)


        ##Decoder pose_r
        self.up1_pose_r = Up(1024, 512 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up2_pose_r = Up(512, 256 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up3_pose_r = Up(256, 128 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up4_pose_r = Up(128, 64, constants.kernel_size, constants.padding_size, bilinear)
        self.outc_pose_r = OutConv(64, constants.pose_r_size, constants.kernel_size, constants.padding_size)

        ##Decoder AUs
        self.up1_au = Up(1024, 512 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up2_au = Up(512, 256 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up3_au = Up(256, 128 // factor, constants.kernel_size, constants.padding_size, bilinear)
        self.up4_au = Up(128, 64, constants.kernel_size, constants.padding_size, bilinear)
        self.outc_au = OutConv(64, constants.au_size, constants.kernel_size, constants.padding_size)


    def forward(self, x):
        #Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        #Decoder gaze
        x = self.up1_eye(x5, x4)
        x = self.up2_eye(x, x3)
        x = self.up3_eye(x, x2)
        x = self.up4_eye(x, x1)
        logits_eye = self.outc_eye(x)
        logits_eye = torch.sigmoid(logits_eye)

        #Decoder pose_t
        x = self.up1_pose_t(x5, x4)
        x = self.up2_pose_t(x, x3)
        x = self.up3_pose_t(x, x2)
        x = self.up4_pose_t(x, x1)
        logits_pose_t = self.outc_pose_t(x)
        logits_pose_t = torch.sigmoid(logits_pose_t)

        #Decoder pose_r
        x = self.up1_pose_r(x5, x4)
        x = self.up2_pose_r(x, x3)
        x = self.up3_pose_r(x, x2)
        x = self.up4_pose_r(x, x1)
        logits_pose_r = self.outc_pose_r(x)
        logits_pose_r = torch.sigmoid(logits_pose_r)

        #Decoder AUs
        x = self.up1_au(x5, x4)
        x = self.up2_au(x, x3)
        x = self.up3_au(x, x2)
        x = self.up4_au(x, x1)
        logits_au = self.outc_au(x)
        logits_au = torch.sigmoid(logits_au)
        
        return logits_eye, logits_pose_t, logits_pose_r, logits_au
