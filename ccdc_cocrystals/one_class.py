# Import the basic libraries
from sklearn import datasets, metrics
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from scipy.spatial.distance import squareform
from matplotlib import cm
import itertools
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import sys
sys.path.extend(['deep_one_class', 'deep_one_class/src/set_transformer', 'deep_one_class.src.base' ])
from deep_one_class.src.set_transformer.modules import SAB, PMA, ISAB
import tqdm
#from deep_one_class.src.base.torchvision_dataset import TorchvisionDataset
import logging
import random
from deep_one_class.src.utils.config import Config
from deep_one_class.src import deepSVDD
from deep_one_class.src.deepSVDD import *
from deep_one_class.src.base.base_net import BaseNet
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import figure
#sys.setrecursionlimit(10**6) 

cfg = Config({'nu': 0.05, 
              'objective':  'one-class'} ) 


def load_model():
    """ 
    Load the deep learning model
    """
    torch.manual_seed(0)
    deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    net_name='model_checkpoint_final.pth'
    deep_SVDD.set_network(net_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deep_SVDD.load_model('deep_one_class/avg_checkpoint.pth')

    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class PairsEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 100
        self.seq = nn.Sequential(SAB(dim_in=4096, dim_out=1000, num_heads=2),
            SAB(dim_in=1000, dim_out=500, num_heads=2),
            SAB(dim_in=500, dim_out=100, num_heads=2),
            PMA(dim=100, num_heads=5, num_seeds=1))
        
    def forward(self, inp):
      x = torch.split(inp,4096, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)

class PairsAutoEncoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=100, out_features=500), nn.LeakyReLU(),
        nn.Linear(in_features=500, out_features=1000),nn.LeakyReLU(),
        nn.Linear(in_features=1000, out_features=1613),nn.LeakyReLU(),         
        nn.Linear(in_features=4096, out_features=8192), nn.Sigmoid())
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))

def build_autoencoder(net_name):
    return PairsAutoEncoder()

def build_network(net_name):  
    return PairsEncoder()
