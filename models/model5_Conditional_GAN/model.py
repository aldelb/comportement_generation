from logging import exception
import torch
import torch.nn as nn
import torch.nn.functional as F
import constant
import contextlib
from torchsummary import summary

def write_model_5(file_path, model, D):
    # probleme d'affichage de summary avec LSTM et GRU
    NotImplemented

def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, constant.kernel_size, padding = constant.padding_size, bias=True),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )   

class Model(nn.Module):
    def __init__(self, h_size, o_size):
        super(Model, self).__init__()

        if(constant.layer == "LSTM"):
            self.layer = nn.LSTM(input_size = h_size, hidden_size = h_size, num_layers=2, bidirectional=True, dropout=constant.dropout)
            self.o_fc = nn.Linear(2 * h_size, o_size)
        elif(constant.layer == "GRU"):
            self.layer = nn.GRU(input_size = h_size, hidden_size = h_size, num_layers=2, bidirectional=True, dropout=constant.dropout)
            self.o_fc = nn.Linear(2 * h_size, o_size)
        else: #CONV
            self.layer1 = conv_bn_relu(h_size, int(h_size/2))
            self.layer2 = conv_bn_relu(int(h_size/2), int(h_size/2))
            self.o_fc = nn.Linear(int(h_size/2), o_size)

    def applyLayer(self, x):
        if(constant.layer == "LSTM" or constant.layer == "GRU"):
            x = torch.reshape(x, (x.shape[2], x.shape[0], x.shape[1]))
            return self.layer(x)[0]
        else: #CONV
            x = self.layer1(x)
            x = self.layer2(x)
            x = torch.reshape(x, (x.shape[2], x.shape[0], x.shape[1]))
            return x

    def formatOutput(self, x):
        return torch.reshape(x, (x.shape[1], x.shape[2], x.shape[0]))

class Generator(Model):

    def __init__(self):
        super(Generator, self).__init__(constant.hidden_size, constant.pose_size + constant.au_size)

        self.h_size = constant.hidden_size
        self.i_size = constant.prosody_size #prosody size
        self.n_size = constant.noise_size #noise size
        self.o_size = constant.pose_size + constant.au_size #openface size

        self.i_fc = nn.Linear(self.i_size, int(self.h_size/2)) #apply linear transformation to incoming data (with additive bias)
        self.n_fc = nn.Linear(self.n_size, int(self.h_size/2))
        
    def forward(self, i, n):
        n = torch.reshape(n, (-1, n.shape[2], n.shape[1]))
        n = n.repeat(1, 1, 300)
        bs, l = i.size(0), i.size(2)
        i = i.view(bs*l, -1)
        n = n.view(bs*l, -1)

        i = F.leaky_relu(self.i_fc(i), 1e-2) #leaky_relu : x if x > 0, negative_slope*x otherwise
        n = F.leaky_relu(self.n_fc(n), 1e-2)

        i = i.view(bs, -1, l)
        n = n.view(bs, -1, l)

        x = torch.cat([i, n], dim=1) #concatenate the tensors on the given dimensions 
        x = self.applyLayer(x)
        x = F.leaky_relu(x, 1e-2)
        x = self.o_fc(x)
        x = self.formatOutput(x)
        return x

class Discriminator(Model):

    def __init__(self):

        super(Discriminator, self).__init__(constant.hidden_size, 1)

        self.i_size = constant.pose_size + constant.au_size
        self.c_size = constant.prosody_size
        self.h_size = constant.hidden_size

        self.i_fc = nn.Linear(self.i_size, int(self.h_size/2))
        self.c_fc = nn.Linear(self.c_size, int(self.h_size/2))

    def forward(self, x, c):
        #c : condition = prosodique features
        #x : real or fake openface data 
        bs, l = x.size(0), x.size(2)
        x = x.view(bs*l, -1)
        c = c.view(bs*l, -1)

        x = F.leaky_relu(self.i_fc(x))
        c = F.leaky_relu(self.c_fc(c))

        x = x.view(bs, -1, l)
        c = c.view(bs, -1, l)

        x = torch.cat([x, c], dim=1)
        x = self.applyLayer(x)
        x = torch.sigmoid(self.o_fc(x))
        x = self.formatOutput(x)
        return x
