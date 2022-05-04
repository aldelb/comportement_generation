import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constant
import contextlib
from torchsummary import summary
#pip install torchsummary

def write_model_2(file_path, model, D):
        with open(file_path, "a") as o:
            with contextlib.redirect_stdout(o):
                summary(model, (constant.prosody_size, 300), batch_size = constant.batch_size)
        o.close()

def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )   

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)  
        self.upsample75 = nn.Upsample(75)  
        
        ##Encoder
        self.conv_down1 = conv_bn_relu(constant.prosody_size, 64)
        self.conv_down2 = conv_bn_relu(64, 128)
        self.conv_down3 = conv_bn_relu(128, 256)
        self.conv_down4 = conv_bn_relu(256, 512)  

        ##Decoder
        self.conv_up3 = conv_bn_relu(256 + 512, 256)
        self.conv_up2 = conv_bn_relu(128 + 256, 128)
        self.conv_up1 = conv_bn_relu(128 + 64, 64)
        self.conv_last = nn.Conv1d(64, constant.pose_size + constant.au_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)


    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.conv_down4(x)
        
        x = self.upsample75(x) 
        x = torch.cat([x, conv3], dim=1)
        
        x = self.conv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.conv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.conv_up1(x)
        
        out = self.conv_last(x)
        
        return out