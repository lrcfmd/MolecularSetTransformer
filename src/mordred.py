import pandas as pd
import numpy as np
import base64
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from . import one_class_model
from src.one_class_model import *
from src.utils.config import Config
from src.base.torchvision_dataset import BaseDataset
from src.utils.config import Config
from src.base.base_net import BaseNet
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import glob
from mordred import Calculator, descriptors  #https://github.com/mordred-descriptor/mordred
calc = Calculator(descriptors, ignore_3D=True)
from src.set_transformer.modules import *


def smile_to_modred(smile):
    return calc(Chem.MolFromSmiles(smile))

def get_modred(smiles):
    paws = []
    for smile in smiles:
        paws.append(np.asarray(smile_to_modred(smile)))
    return paws

def get_modred_df(smiles):
    df = pd.DataFrame(get_modred(smiles))
    df = pd.DataFrame(df.values, columns=[str(d) for d in calc.descriptors])
    return df

def get_mordred_representation(dataset):
    """ Given the smiles of a validation dataset get the mordred descriptors """   
    df = pd.concat([dataset['Name1'], dataset['Name2'], get_modred_df(dataset['Smiles1'].values),
     get_modred_df(dataset['Smiles2'].values)], axis=1)
    return df

class Pairs_Dataset(BaseDataset):

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


class PairsEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 50
        self.seq = nn.Sequential(SAB(dim_in=1613, dim_out=500, num_heads=10),
              SAB(dim_in=500, dim_out=100, num_heads=10),
              SAB(dim_in=100, dim_out=50, num_heads=10),
              PMA(dim=50, num_heads=10, num_seeds=1),
            PMA(dim=50, num_heads=5, num_seeds=1))
        
    def forward(self, inp):
      x = torch.split(inp, 1613, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)

class PairsAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=50, out_features=1613), nn.LeakyReLU(),
        nn.Sequential(nn.Linear(in_features=1613, out_features=3226), nn.Sigmoid()))
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_dataset(filename):
    dataset=pd.read_csv(filename, encoding='latin1')
    dataset = dataset.iloc[:10,:]
    smiles1 =  dataset['smiles1']
    smiles2 =  dataset['smiles2']
    df_mordred = pd.concat([pd.DataFrame(get_modred_df(dataset.smiles1).apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0)), columns=[str(d) for d in calc.descriptors]),
     pd.DataFrame(get_modred_df(dataset.smiles2).apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0)), columns=[str(d) for d in calc.descriptors])],axis=1)
    dataset = Pairs_Dataset('', data= df_mordred.iloc[:, :] )
    return dataset

