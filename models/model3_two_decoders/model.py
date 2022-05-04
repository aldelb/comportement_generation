import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constant
import contextlib
from torchsummary import summary
#pip install torchsummary

def write_model_3(file_path, model, D):
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
        ##Encoder
        self.conv_down1 = conv_bn_relu(constant.prosody_size, 64)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv_down2 = conv_bn_relu(64, 128)
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv_down3 = conv_bn_relu(128, 256)
        self.maxpool3 = nn.MaxPool1d(2)
        self.conv_down4 = conv_bn_relu(256, 512)  

        
        ##Decoder pose
        self.upsample75_pose = nn.Upsample(75) 
        self.conv_up3_pose = conv_bn_relu(256 + 512, 256)
        self.upsample3_pose = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)  
        self.conv_up2_pose = conv_bn_relu(128 + 256, 128)
        self.upsample2_pose = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)  
        self.conv_up1_pose = conv_bn_relu(128 + 64, 64) 
        
        self.conv_last_pose = nn.Conv1d(64, constant.pose_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)

        ##Decoder AUs
        self.upsample75_au = nn.Upsample(75)  
        self.conv_up3_au = conv_bn_relu(256 + 512, 256)
        self.upsample3_au = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)  
        self.conv_up2_au = conv_bn_relu(128 + 256, 128)
        self.upsample2_au = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)  
        self.conv_up1_au = conv_bn_relu(128 + 64, 64) 

        self.conv_last_au = nn.Conv1d(64, constant.au_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)

    def forward(self, x):
        #Encoder
        conv1 = self.conv_down1(x)
        x = self.maxpool1(conv1)

        conv2 = self.conv_down2(x)
        x = self.maxpool2(conv2)
        
        conv3 = self.conv_down3(x)
        x = self.maxpool3(conv3)
        
        last_x_encoder = self.conv_down4(x)
        
        #Decoder pose and gaze angle
        x = self.upsample75_pose(last_x_encoder)
        x = torch.cat([x, conv3], dim=1)

        x = self.conv_up3_pose(x)
        x = self.upsample3_pose(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.conv_up2_pose(x)
        x = self.upsample2_pose(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.conv_up1_pose(x)
        
        out_pose = self.conv_last_pose(x)

        #Decoder AUs
        x = self.upsample75_au(last_x_encoder)
        x = torch.cat([x, conv3], dim=1)

        x = self.conv_up3_au(x)
        x = self.upsample3_au(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.conv_up2_au(x)
        x = self.upsample2_au(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.conv_up1_au(x)
        
        out_au = self.conv_last_au(x)
        
        return out_pose, out_au