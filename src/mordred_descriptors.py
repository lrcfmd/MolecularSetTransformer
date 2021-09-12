import pandas as pd
import numpy as np
import base64
import os
import torch 
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import glob
from sklearn.preprocessing import MinMaxScaler
from deep_one_class.src.optim.ae_trainer import bidirectional_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from deep_one_class.src import deepSVDD
from deep_one_class.visualizations import plots
from deep_one_class.src.deepSVDD import *
from deep_one_class.src.utils.config import Config
from mordred import Calculator, descriptors  #https://github.com/mordred-descriptor/mordred
calc = Calculator(descriptors, ignore_3D=True)
import joblib


cfg = Config({'nu': 0.05, 
              'objective':  'one-class'} ) 

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

def ae_score(deep_SVDD, X):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        net = deep_SVDD.ae_net.to(device)
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        scores = bidirectional_score(X, y)
    return scores

    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


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

def build_autoencoder(net_name):
    return PairsAutoEncoder()

def build_network(net_name):  
    return PairsEncoder()

def score(deep_SVDD, X):
    with torch.no_grad():
        device = 'cpu'  
        net = deep_SVDD.net.to(device)
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        c, R = torch.FloatTensor([deep_SVDD.c]).to(device), torch.FloatTensor([deep_SVDD.R]).to(device)
        dist = torch.sum((y - c)**2, dim=1)
        if deep_SVDD.objective == 'soft-boundary':
            scores = dist - R ** 2
        else:
            scores = dist
    return scores

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-dataset_name', required=True, type = str, help = "Dataset Name", default = None)
    parser.add_argument('-save_dir', required=False, type = str, help = "Directory to save your output", default = None)
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset_name, encoding='latin1') 
    names = pd.concat([dataset.Name1, dataset.Name2], axis=1)
    dataset=dataset.iloc[:,:]
    label= dataset['Cocrystal']
    X_scaler = joblib.load("deep_one_class/saved_models/mordred_train_scaler_cpu.save") 
    validation_set = pd.concat([pd.DataFrame(X_scaler.transform(get_modred_df(dataset.Smiles1).apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)), columns=[str(d) for d in calc.descriptors]),
    pd.DataFrame(X_scaler.transform(get_modred_df(dataset.Smiles2).apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)), columns=[str(d) for d in calc.descriptors])],axis=1)
    torch.manual_seed(0)
    deepSVDD.build_network = build_network
    deepSVDD.build_autoencoder = build_autoencoder
    deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    net_name='mordred_model.pth'
    deep_SVDD.set_network(net_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    deep_SVDD.load_model('deep_one_class/mordred_model_cpu.pth', True)
    scores = -1*ae_score(deep_SVDD, validation_set.iloc[:,:].values).cpu().detach().numpy()
    scaler = MinMaxScaler()
    scores_scaled = scaler.fit_transform(scores.reshape(-1,1)).ravel()
    dataset['scores'] = scores_scaled
    dataset_name = args.dataset_name.split('/')[-1].split('.')[0]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dataset.to_csv(f'{args.save_dir}/{dataset_name}.csv') #'data/artemisin_with_scores.csv')  # set the name of the file for saving the scores
    roc_auc = roc_auc_score(label.values, scores)
    print('total ROC-AUC', roc_auc)
    plots.ranking_plot(scores_scaled, label, names.values, args.save_dir)
    plots.plot_confusion_matrix(label ,scores_scaled , args.threshold, args.save_dir) # here you can select your threshold for calculating the confusion matrix (usually > 0.8)

if __name__ == "__main__":     
    main()