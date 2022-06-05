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
from src import morgan
from src import gnn
from src import mordred
from src import chemberta
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

def get_dataset(model, file_name):
    if model == 'gnn':
        return gnn.load_dataset(file_name)
    elif model == 'chemberta':
        return chemberta.load_dataset(file_name)
    elif model == 'morgan':
        return morgan.load_dataset(file_name)
    return mordred.load_dataset(file_name)

def set_neural_network(model_name):
    if model_name == 'gnn':
        PairsAutoEncoder, PairsEncoder = gnn.PairsAutoEncoder, gnn.PairsEncoder
    elif model_name == 'chemberta':
        PairsAutoEncoder, PairsEncoder = chemberta.PairsAutoEncoder, chemberta.PairsEncoder
    elif model_name == 'morgan':
        PairsAutoEncoder, PairsEncoder = morgan.PairsAutoEncoder, chemberta.PairsEncoder
    elif model_name == 'mordred':
        PairsAutoEncoder, PairsEncoder = mordred.PairsAutoEncoder, mordred.PairsEncoder

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

def wandb_sweep(dataset, config):
    import wandb
    config_defaults = {
        'n_epochs': 150,
        'batch_size': 32,
        'weight_decay': 1e-04,
        'lr': 1e-4,
        'optimizer_name': 'adam',
    }
    wandb.init(config=config_defaults)
    config = wandb.config
    deep_SVDD.pretrain(dataset,
                   optimizer_name='adam',
                   lr=config.lr,
                   n_epochs = config.n_epochs , #250, 1e-4,32,1e-04
                   lr_milestones=(100,),
                   batch_size= config.batch_size , 
                   weight_decay= config.weight_decay ,#cfg.settings['ae_weight_decay'],  
                   device=device,
                   n_jobs_dataloader=0, use_wandb=True)
    wandb.log({"auc": deep_SVDD.ae_trainer.test(dataset, deep_SVDD.ae_net)}) 
    wandb.agent(sweep_id, train)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default= 'morgan', choices = ['gnn', 'morgan', 'chemberta', 'mordred'] , help='The model to use for training')
    parser.add_argument('--training_data', help='The csv file with the training smiles pairs')
    parser.add_argument('--save_dir', help='The directory to save the trained network')
    parser.add_argument('-n_epochs', default= 50, type = int, help='Set the number of epochs')
    parser.add_argument('-lr', default= 0.001, type = float, help='Set the learning rate')
    parser.add_argument('--use_wandb', action='store_true' ,help='Set the learning rate')
    args = parser.parse_args()

    set_seed()
    deep_SVDD = deepSVDD.DeepSVDD('one-class', 0.05)
    #deep_SVDD.set_network('Molecular Set Transformer')
    dataset = get_dataset(args.model, args.training_data)
    set_neural_network(args.model)

    if args.use_wandb:
        import wandb
        config_defaults = {
        'n_epochs': 50,
        'batch_size': 32,
        'weight_decay': 1e-04,
        'lr': 1e-4,
        'optimizer_name': 'adam',
        }
        wandb.init(config=config_defaults)
        config = wandb.config


    deep_SVDD.pretrain(dataset,
                    optimizer_name='adam',
                    lr= args.lr,
                    n_epochs = args.n_epochs, 
                    lr_milestones=(100,),
                    batch_size= 32, 
                    weight_decay= 0.00001,
                    device= 'cpu',
                    n_jobs_dataloader=0)
    



    deep_SVDD.save_model(f'{args.save_dir}/model.pth')

if __name__ == "__main__":
    main()
