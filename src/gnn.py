import pandas as pd
import numpy as np
import base64
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from src import deepSVDD
from src.deepSVDD import *
from src.utils.config import Config
from src.base.torchvision_dataset import TorchvisionDataset
from src.utils.config import Config
from src.base.base_net import BaseNet
import subprocess

def get_representation(smiles1, smiles2):
    """ Given the smiles of a validation dataset convert it to fingerprint
     representation """   
    df = pd.concat([pd.DataFrame(smiles1, columns=['smiles1']),
     pd.DataFrame(smiles2, columns=['smiles2'])], axis=1)
    return df

def smiles2txt(dataset):  
    ''' reading the smiles from the csv file and saves them in txt file in order to get the 
    graph embendings from each smile'''

    with open(os.path.join("src\\gnn",'smiles1.txt'), 'w') as f:
        for item in dataset['smiles1'].values:
            f.write("%s\n" % item)

    with open(os.path.join("src\\gnn", 'smiles2.txt'), 'w') as f:
        for item in dataset['smiles2'].values:
            f.write("%s\n" % item)

class Pairs_Dataset(TorchvisionDataset):

    def __init__(self, root: str, train_idx=None, test_idx=None, data=None):
        super().__init__(root)
        ## Loading the train set
        self.train_set = Pairs(root=self.root, train=True, data=data)
        if train_idx is not None:
          self.train_set = Subset(self.train_set, train_idx)
        ## Loading the test set
        self.test_set = Pairs(root=self.root, train=False, data=data)
        if test_idx is not None:
            self.test_set = Subset(self.test_set, test_idx)

class Pairs(Dataset):

    def __init__(self, root, train, data=None):
        super(Pairs, self).__init__()
        self.train = train
        # Setup the train dataset
        self.data=data.values.astype('f')
        self.labels=np.zeros(self.data.shape[0])

    # This is used to return a single datapoint. A requirement from pytorch
    def __getitem__(self, index):
        return self.data[index], self.labels[index], index

    # For Pytorch to know how many datapoints are in the dataset
    def __len__(self):
        return len(self.data)
        return self.decoder(self.encoder(x))


class PairsEncoder(nn.Module):

    def __init__(self,proba=0.1):
        super().__init__()
        self.rep_dim = 50
        self.seq = nn.Sequential(SAB(dim_in=300, dim_out=150, num_heads=5),
                                 nn.Dropout(p=proba),
            SAB(dim_in=150, dim_out=50, num_heads=5),
        PMA(dim=50, num_heads=2, num_seeds=1))

    def forward(self, inp):
      x = torch.split(inp, 300, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)


class PairsAutoEncoder(nn.Module):
    def __init__(self, proba=0.1):
        super().__init__()
        self.encoder = PairsEncoder(proba)
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=50, out_features=300), nn.LeakyReLU(),
                                     #nn.Dropout(p=proba),
        nn.Linear(in_features=300, out_features=600))
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x)).squeeze()


def load_dataset(filename):
    dataset=pd.read_csv(filename, encoding='latin1')
    dataset = dataset.iloc[:10,:]
    smiles1 =  dataset['smiles1']
    smiles2 =  dataset['smiles2']
    validation_set= get_representation(smiles1, smiles2)    
    smiles2txt(validation_set)
    python = sys.executable
    subprocess.call(f"{python} src/gnn/main.py -fi src/gnn/smiles1.txt -m gin_supervised_masking -o src/gnn/results1", shell=True)
    subprocess.call(f"{python} src/gnn/main.py -fi src/gnn/smiles2.txt -m gin_supervised_masking -o src/gnn/results2", shell=True)
    valid1 = np.load('src/gnn/results1/mol_emb.npy')
    valid2 = np.load('src/gnn/results2/mol_emb.npy')
    df_gnn = pd.concat([pd.DataFrame(valid1), pd.DataFrame(valid2)],axis=1)   
    dataset = Pairs_Dataset('', data= df_gnn.iloc[:, :] )
    return dataset