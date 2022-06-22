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
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs
from src.set_transformer.modules import *

model_args = ModelArgs(max_seq_length=156)
model = RepresentationModel(
        model_type="roberta",
        model_name="seyonec/PubChem10M_SMILES_BPE_396_250",
        use_cuda=False)

def mean_pool(model, sentences):
  attn_mask = model._tokenize(sentences)['attention_mask'].numpy()
  word_vectors = model.encode_sentences(sentences, combine_strategy=0)
  return word_vectors

def smiles2txt(dataset):  
    ''' reading the smiles from the csv file and saves them in txt file in order to get the 
    graph embendings from each smile'''

    with open(os.path.join("src\\gnn",'smiles1.txt'), 'w') as f:
        for item in dataset['smiles1'].values:
            f.write("%s\n" % item)

    with open(os.path.join("src\\gnn", 'smiles2.txt'), 'w') as f:
        for item in dataset['smiles2'].values:
            f.write("%s\n" % item)

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
        self.seq = nn.Sequential(SAB(dim_in=768, dim_out=150, num_heads=5),
              SAB(dim_in=150, dim_out=50, num_heads=5),
            PMA(dim=50, num_heads=2, num_seeds=1))
        
    def forward(self, inp):
      x = torch.split(inp, 768, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)

class PairsAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=50, out_features=768), nn.LeakyReLU(),
                          #           nn.Linear(in_features=150, out_features=300), nn.LeakyReLU(),
        nn.Linear(in_features=768, out_features=1536))
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_dataset(filename):    
    dataset=pd.read_csv(filename, encoding='latin1')
    dataset = dataset.iloc[:10,:]
    smiles1 =  dataset['smiles1']
    smiles2 =  dataset['smiles2']
    train_smiles1 = smiles1.values
    train_smiles1_vectors = mean_pool(model, train_smiles1)
    train_smiles2 = smiles2.values
    train_smiles2_vectors = mean_pool(model, train_smiles2)
    df_chemberta = pd.concat([pd.DataFrame(train_smiles1_vectors),pd.DataFrame(train_smiles2_vectors)], axis=1)
    dataset = Pairs_Dataset('', data= df_chemberta.iloc[:, :] )
    return dataset