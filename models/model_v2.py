import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def deconv_block(in_chans,out_chans,kernel_size=3,stride=3,padding=0):
        return nn.Sequential(nn.ConvTranspose2d(in_channels=in_chans,out_channels=out_chans,kernel_size=kernel_size,stride=stride,padding=padding)
                             ,nn.BatchNorm2d(out_chans),nn.ReLU())

def deconv_block_final(in_chans,out_chans,kernel_size=3,stride=3,padding=0):
        return nn.Sequential(nn.ConvTranspose2d(in_channels=in_chans,out_channels=out_chans,kernel_size=kernel_size,stride=stride,padding=padding))

def conv_block(in_chan,out_chan):
    return nn.Sequential(nn.Conv2d(in_channels=in_chan,out_channels=out_chan,kernel_size=3,padding=1),
                         nn.MaxPool2d(2,2),nn.BatchNorm2d(out_chan),nn.ReLU())

def linear_block(in_dim,out_dim):
    return nn.Sequential(nn.Linear(in_dim,out_dim),nn.BatchNorm1d(out_dim),nn.ReLU())

class reshape(nn.Module):
    def __init__(self,shape):
        super(reshape,self).__init__()
        self.shape = shape
    def forward(self,x):
        return x.view(self.shape)

class Enc(nn.Module):
    def __init__(self,ncond,latent_dim):
        super(Enc,self).__init__()
        self.ncond = ncond
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(conv_block(2,16),conv_block(16,32),conv_block(32,64),conv_block(64,128),Flatten(),
                                linear_block(3456,128))
        self.z_enc = nn.Linear(128+ncond,16)

#L1 = conv_block(2,16)
#L2 = conv_block(16,32)
#L3 = conv_block(32,64)
#L4 = conv_block(64,128)
#L5 = Flatten()
#L6 = nn.Linear(3456,128)
#L7 = nn.Linear(128+ncond,16)

    def forward(self,x,y):
        x = self.enc(x)
        x = self.z_enc(torch.cat((x,y),1))
        return x

class Dec(nn.Module):
    def __init__(self,ncond,latent_dim):
        super(Dec,self).__init__()
        self.ncond = ncond
        self.latent_dim = latent_dim
        self.dec = nn.Sequential(linear_block(16,128),linear_block(128,3456),reshape((-1,128,3,9)),
                                deconv_block(128,64,4,2,padding=(1,1)),deconv_block(64,32,4,2,padding=(1,1)),
                                deconv_block(32,16,4,2,padding=(1,1)),deconv_block_final(16,2,4,2,padding=(1,1)))
        self.out_act_hits = nn.Sigmoid()
        self.out_act_time = nn.ReLU()

#d1 = linear_block(16,128)
#d2 = linear_block(128,3456)
#b.view(-1,128,3,9)
#d3 = deconv_block(128,64,4,2,padding=(1,1))
#d4 = deconv_block(64,32,4,2,padding=(1,1))
#d5 = deconv_block(32,16,4,2,padding=(1,1))
#d6 = deconv_block(16,2,4,2,padding=(1,1))

    def forward(self,x):
        x = self.dec(x)
        shape = (-1,1,48,144)
        x1 = self.out_act_hits(x[:,0,:,:]).view(shape)
        x2 = self.out_act_time(x[:,1,:,:]).view(shape)
        x = torch.cat((x1,x2),1)
        return x
'''
class CNN_MLP(nn.Module):
    def __init__(self,shape):
        super(CNN_MLP,self).__init__()
        self.shape = shape
        self.cnn = nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3),
                torch.nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3),
                torch.nn.BatchNorm1d(256),
                nn.ReLU(),
                Flatten(),
                nn.Dropout(0.5),
            )
        self.mlp = nn.Sequential(nn.Linear(3072,100),
                                 nn.ReLU(),
                                 nn.Linear(100,50),
                                 nn.ReLU(),
                                 nn.Linear(50,25),
                                 nn.ReLU(),
                                 nn.Linear(25,1),
                                 nn.Sigmoid())

    def forward(self,x):
        out = self.cnn(x.unsqueeze(1))
        out = self.mlp(out)
        return out
'''

class CNN_MLP(nn.Module):
    def __init__(self,shape):
        super(CNN_MLP,self).__init__()
        self.shape = shape
        self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3),
                torch.nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3),
                torch.nn.BatchNorm1d(32),
                nn.ReLU(),
                Flatten(),
            )
        self.mlp = nn.Sequential(nn.Linear(384,100),
                                 nn.Dropout(0.50),
                                 nn.BatchNorm1d(100),
                                 nn.ReLU(),
                                 nn.Linear(100,50),
                                 nn.Dropout(0.50),
                                 nn.BatchNorm1d(50),
                                 nn.ReLU(),
                                 nn.Linear(50,25),
                                 nn.Dropout(0.50),
                                 nn.BatchNorm1d(25),
                                 nn.ReLU(),
                                 nn.Linear(25,1),
                                 nn.Sigmoid())

    def forward(self,x):
        out = self.cnn(x.unsqueeze(1))
        out = self.mlp(out)
        return out

class cVAE_DIRC(nn.Module):
    def __init__(self,nhid = 16, ncond = 5,classification=0):
        super(cVAE_DIRC, self).__init__()
        self.dim = nhid
        self.encoder = Enc(ncond,nhid)
        self.decoder = Dec(ncond,nhid)
        self.classification = bool(classification)
        if self.classification:
            self.CNN_MLP = CNN_MLP(1)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y):
        #y = self.label_embedding(y)
        z = self.encoder(x, y)
        if self.classification:
            pred = self.CNN_MLP(z)
        #z = self.sampling(mean, logvar)
            return self.decoder(z), z,pred
        else:
            return self.decoder(z),z
