import sys
import json
import os.path
import torch 
from src import one_class_model
from src.one_class_model import *
from src import morgan
from src import gnn
from src import mordred
from src import chemberta
import argparse

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
    one_class_model.build_network = build_network
    one_class_model.build_autoencoder = build_autoencoder

    return PairsAutoEncoder()

def load_model(model_type, model_path):
    ae = set_neural_network(model_type)
    model_dict = torch.load(model_path,map_location='cpu')
    ae.load_state_dict(model_dict['ae_net_dict'])
    return ae