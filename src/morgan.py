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
from rdkit import Chem
from rdkit.Chem import AllChem

def get_representation(dataset):
    """ Given the smiles of a validation dataset convert it to fingerprint
     representation """   
    df = pd.concat([fingerprint_from_df(dataset['smiles1'].values, 'paws_1'),
     fingerprint_from_df(dataset['smiles2'].values, 'paws_2')], axis=1)
    return df

def smile_to_fingerprint(smile):
  mol = Chem.MolFromSmiles(smile)
  return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096, useChirality=True)

def fingerprint(smiles):
  bits = []
  for smile in smiles:
    bits.append(np.asarray(smile_to_fingerprint(smile)))
  return bits

def fingerprint_from_df(smiles, prefix):
  df = pd.DataFrame(fingerprint(smiles))
  columns = [f'{prefix}_{i}' for i in df.columns]
  df.columns = columns
  return df

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


class PairsEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 50
        self.seq = nn.Sequential(SAB(dim_in=4096, dim_out=500, num_heads=10),
              #SAB(dim_in=1500, dim_out=500, num_heads=2),
              SAB(dim_in=500, dim_out=50, num_heads=10),
            PMA(dim=50, num_heads=10, num_seeds=1))
        
    def forward(self, inp):
      x = torch.split(inp, 4096, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)

class PairsAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=50, out_features=4096), nn.LeakyReLU(),
        nn.Linear(in_features=4096, out_features=8192), nn.Sigmoid())
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_dataset(filename):
    dataset=pd.read_csv(filename)
    dataset = dataset.iloc[:10,:]
    df_morgan = get_representation(dataset)
    dataset = Pairs_Dataset('', data= df_morgan.iloc[:, :] )
    return dataset