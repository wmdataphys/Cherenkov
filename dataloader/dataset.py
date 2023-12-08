from torch.utils.data import Dataset
import numpy as np
import os
import torch
import random
import collections

# class DIRC_Dataset(Dataset):
#
#     def __init__(self,root_dir,data_file,cond_file,label_file):
#
#         self.root_dir = root_dir
#         self.data_file = os.path.join(self.root_dir,data_file)
#         self.cond_file = os.path.join(self.root_dir,cond_file)
#         self.label_file = os.path.join(self.root_dir,label_file)
#
#         self.data = np.load(self.data_file,allow_pickle=True)
#         self.conds = np.load(self.cond_file,allow_pickle=True)
#         self.labels = np.load(self.label_file,allow_pickle=True)
#
#         # Preshuffled. Want to pass indices to Subset instead.
#         #temp = list(zip(self.data,self.conds,self.labels))
#         #random.shuffle(temp)
#         #self.data,self.conds,self.labels = zip(*temp)
#         #self.data,self.conds,self.labels = list(self.data),list(self.conds),list(self.labels)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#
#         # Get the sample
#         max_time = 500.
#         min_time = 0.003855
#
#         max_P = 11.987511
#         min_P = 0.274731
#
#         max_theta = 12.688103
#         min_theta = 0.93969
#
#         max_phi = 174.967754
#         min_phi = -174.563288
#
#         max_track_x = 105.417099
#         min_track_x = -103.833328
#
#         max_track_y = 102.406105
#         min_track_y = -101.743774
#
#         hits = self.data[index]
#         row,col,time = hits[:,0].astype('int'),hits[:,1].astype('int'),hits[:,2]
#         time = np.exp(time)
#
#         row = row[np.where(time < 500.)]
#         col = col[np.where(time < 500.)]
#         time = time[np.where(time < 500.)]
#         #Xsc=X−XminXmax−Xmin.
#         time = (time - min_time)/(max_time - min_time)
#         PID = self.labels[index]
#         # P, Theta, Phi, Track X, Track Y
#         cond = self.conds[index]
#
#         cond[0] = (cond[0] - min_P)/(max_P - min_P)
#         cond[1] = (cond[1] - min_theta)/(max_theta - min_theta)
#         cond[2] = (cond[2] - min_phi)/(max_phi - min_phi)
#         cond[3] = (cond[3] - min_track_x)/(max_track_x - min_track_x)
#         cond[4] = (cond[4] - min_track_y)/(max_track_y - min_track_y)
#
#         # Transform the hits to replicate the Optical Box. We set log(time) as additional channel axis. THIS IS OLD. Use e(t) and min max scale.
#         # Delete any hits with time > 500 for now. Seems OOD.
#
#         assert len(row) == len(time)
#
#         a = np.zeros((1,48,144))
#         a[0,row,col] = 1.0
#         b = np.zeros((1,48,144))
#         b[0,row,col] = time
#
#         optical_box = np.concatenate([a,b],axis=0)
#
#         # Set 211 == 1
#         # Set 321 == 0
#
#         if abs(PID) == 211:
#             PID = 1
#         else:
#             PID = 0
#
#
#         return optical_box,cond,PID

class DIRC_Dataset(Dataset):

    def __init__(self,data,stats=None):
        self.data = data
        self.stats = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # ['EventID','PDG','NHits','BarID','P','Theta','Phi','X','Y','Z',
        # 'pmtID','pixelID','channel','pos_x','pos_y','pos_z','leadTime']

        # Get the sample
        data = self.data[idx]

        PID = data['PDG']
        cond = np.array([data['P'],data['Theta'],data['Phi'],data['X'],data['Y']])

        pmtID = np.array(data['pmtID'])
        o_box = pmtID//108

        if o_box[0] == 1:
            pmtID -= 108

        if len(np.unique(o_box)) != 1:
            print(np.unique(o_box))

        pixelID = np.array(data['pixelID'])

        row = (pmtID//18) * 8 + pixelID//8
        col = (pmtID%18) * 8 + pixelID%8

        time = np.array(data['leadTime'])

        pos_time = np.where(time > 0)
        row = row[pos_time]
        col = col[pos_time]
        time = time[pos_time]

        if self.stats is not None:
            # Min Max Scale over (0,1)
            cond = (cond - self.stats['cond']['mins'])/(self.stats['cond']['maxes'] - self.stats['cond']['mins'])
            time = np.log(time)


        # Transform the hits to replicate the Optical Box. We set log(time) as additional channel axis. THIS IS OLD. Use e(t) and min max scale.
        # Delete any hits with time > 500 for now. Seems OOD.

        assert len(row) == len(time)

        a = np.zeros((1,48,144))
        a[0,row,col] = 1.0
        b = np.zeros((1,48,144))
        b[0,row,col] = time

        optical_box = np.concatenate([a,b],axis=0)

        # Set 211 == 1
        # Set 321 == 0

        if abs(PID) == 211:
            PID = 1
        else:
            PID = 0


        return optical_box,cond,PID

def create_dataset(config):
    stats = collections.defaultdict(dict)
    stats['cond']['maxes'] = np.array(config['dataset']['stats']['cond_maxes'])
    stats['cond']['mins'] = np.array(config['dataset']['stats']['cond_mins'])
    rho = np.load(config['dataset']['rho_filename'],allow_pickle=True)
    phi = np.load(config['dataset']['phi_filename'],allow_pickle=True)

    data = rho + phi

    random.shuffle(data)

    return DIRC_Dataset(data,stats)
