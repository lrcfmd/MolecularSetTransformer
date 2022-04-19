#python train.py --model gnn --training_data csd_data/csd_cocrystals2020.csv --save_dir src -n_epochs 1 -lr 0
from random import choices
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import os.path
import csv
from collections import Counter
from itertools import groupby
import argparse
import time
#import morgan
from src import gnn
#import mordred
#import chemberta
import random
import torch 
from rdkit import Chem
from rdkit.Chem import AllChem
import glob
from src.optim.ae_trainer import bidirectional_score
from src import deepSVDD
from src.deepSVDD import *
from src.utils.config import Config
from src.utils import *

def get_dataset(dataset_format, file_name):
    if dataset_format == 'gnn':
        return gnn.load_dataset(file_name)
    #elif dataset_format == 'chemberta':
    #    return chemberta.load_dataset(file_name)
    #return mordred.load_dataset(file_name)

def set_neural_network(model_name):
    if model_name == 'gnn':
        PairsAutoEncoder, PairsEncoder = gnn.PairsAutoEncoder, gnn.PairsEncoder
    #elif model_name == 'mordred':
    #    ...
    
    def build_autoencoder(net_name):
        return PairsAutoEncoder()

    def build_network(net_name):  
        return PairsEncoder()

    deepSVDD.build_network = build_network
    deepSVDD.build_autoencoder = build_autoencoder

def set_seed():
  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default= 'morgan', choices = ['gnn', 'morgan', 'chemberta', 'mordred'] , help='The model to use for training')
    parser.add_argument('--training_data')#, default= 'morgan', action='store_true', help='Use the Morgan fingerprint')
    parser.add_argument('--save_dir')#, default= 'morgan', action='store_true', help='Use the Morgan fingerprint')
    parser.add_argument('-n_epochs', default= 50, type =  int, help='Set the number of epochs')
    parser.add_argument('-lr', default= 0.001, type = float, help='Set the learning rate')
    args = parser.parse_args()

    set_seed()
    deep_SVDD = deepSVDD.DeepSVDD('one-class', 0.05)
    deep_SVDD.set_network('Molecular Set Transformer')
    dataset = get_dataset(args.model, args.training_data)
    set_neural_network(args.model)
    deep_SVDD.pretrain(dataset,
                   optimizer_name='adam',
                   lr= args.lr,
                   n_epochs = args.n_epochs, 
                   lr_milestones=(100,),
                   batch_size= 32, 
                   weight_decay= 0.00001,
                   device= 'cpu',
                   n_jobs_dataloader=0)
    deep_SVDD.save_model(args.save_dir)

if __name__ == "__main__":
    main()