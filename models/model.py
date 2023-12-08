import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.ReLU(inplace = True),
                                    nn.Conv2d(16, 32, 5, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.MaxPool2d(2, 2),
                                    Flatten(), MLP([ww*hh*64, 256, 128])
                                   )
        self.calc_mean = MLP([128+ncond, 64, nhid], last_activation = False)
        #self.calc_logvar = MLP([128+ncond, 64, nhid], last_activation = False)
    def forward(self, x, y = None):
        x = self.encode(x)
        if (y is None):
            return self.calc_mean(x)#, self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1))#, self.calc_logvar(torch.cat((x, y), dim=1))

class Decoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid+ncond, 64, 128, 256, c*w*h], last_activation = False), nn.Sigmoid())
    def forward(self, z, y = None):
        c, w, h = self.shape
        if (y is None):
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)

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


class cVAE_DIRC(nn.Module):
    def __init__(self, shape, nclass, nhid = 16, ncond = 16,classification=0):
        super(cVAE_DIRC, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid, ncond = ncond)
        self.decoder = Decoder(shape, nhid, ncond = ncond)
        #self.label_embedding = nn.Embedding(nclass, ncond)
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
            return self.decoder(z, y), z,pred

        else:
            return self.decoder(z,y),z
