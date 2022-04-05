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
#from mordred import Calculator, descriptors  #https://github.com/mordred-descriptor/mordred
calc = Calculator(descriptors, ignore_3D=True)
import joblib
import pickle
import subprocess

cfg = Config({'nu': 0.05, 
              'objective':  'one-class'} ) 

def get_representation(dataset):
    """ Given the smiles of a validation dataset convert it to fingerprint
     representation """  
    wrong_smiles = [] 
    for i in dataset['Smiles1'].values:
        if Chem.MolFromSmiles(i) == None:
            print('Wrong smiles1', i)
            wrong_smiles.append(i)
    for i in dataset['Smiles2'].values:
        if Chem.MolFromSmiles(i) == None:
            print('Wrong smiles2', i)
            wrong_smiles.append(i)
    pd.DataFrame(wrong_smiles, columns=['wrong smiles']).to_csv('wrong_smiles.csv')
    dataset = dataset[~dataset.Smiles1.isin(wrong_smiles)]
    dataset = dataset[~dataset.Smiles2.isin(wrong_smiles)]
    #df = pd.concat([pd.DataFrame(dataset.Cocrystal.values,columns=['Cocrystal']) ,pd.DataFrame(dataset['Name1'].values,columns=['Name1']), 
    #pd.DataFrame(dataset['Smiles1'].values, columns=['Smiles1'] ),
    #pd.DataFrame(dataset['Name2'].values,columns=['Name2']), pd.DataFrame(dataset['Smiles2'].values, columns=['Smiles2'])], axis=1)
    df = pd.concat([pd.DataFrame(dataset.Cocrystal.values,columns=['Cocrystal']) , 
    pd.DataFrame(dataset['Smiles1'].values, columns=['Smiles1'] ),
    pd.DataFrame(dataset['Smiles2'].values, columns=['Smiles2'])], axis=1)
    return df, wrong_smiles

def smiles2txt(dataset):  
    ''' reading the smiles from the csv file and saves them in txt file in order to get the 
    graph embendings from each smile'''

    with open(os.path.join("gnn",'smiles1.txt'), 'w') as f:
        for item in dataset['Smiles1'].values:
            f.write("%s\n" % item)

    with open(os.path.join("gnn", 'smiles2.txt'), 'w') as f:
        for item in dataset['Smiles2'].values:
            f.write("%s\n" % item)

def ae_score(deep_SVDD, X):
    device =  'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.seq = nn.Sequential(SAB(dim_in=300, dim_out=150, num_heads=5),
              SAB(dim_in=150, dim_out=50, num_heads=5),
            PMA(dim=50, num_heads=2, num_seeds=1))
        
    def forward(self, inp):
      x = torch.split(inp, 300, dim=1)     
      x= torch.stack(x).transpose(0,1)
      x = self.seq(x).squeeze()
      return x.view(inp.size(0), -1)

    def get_attention_weights(self):
        return [layer.get_attention_weights() for layer in self.seq]

class PairsAutoEncoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.encoder = PairsEncoder()
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential( nn.Linear(in_features=50, out_features=150), nn.LeakyReLU(),
                                     nn.Linear(in_features=150, out_features=300), nn.LeakyReLU(),
        nn.Linear(in_features=300, out_features=600))
        self.decoder.apply(init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_attention_weights(self):
        return self.encoder.get_attention_weights()

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
    parser.add_argument('-threshold', required=False, type=float , help = "Threshold to generate the confusion matrix", default = 0.8)
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset_name, encoding='latin1') 
    
    dataset=dataset.iloc[:,:]  
    validation_set, wrong_smiles = get_representation(dataset)    
    dataset=dataset[~dataset.Smiles1.isin(wrong_smiles)]
    dataset=dataset[~dataset.Smiles2.isin(wrong_smiles)]
    names = pd.concat([dataset.Name2, dataset.Name2], axis=1)
    label= validation_set['Cocrystal']
    smiles2txt(dataset)
    subprocess.call("python gnn/main.py -fi gnn/smiles1.txt -m gin_supervised_masking -o gnn/results1", shell=True)
    subprocess.call("python gnn/main.py -fi gnn/smiles2.txt -m gin_supervised_masking -o gnn/results2", shell=True)
    valid1 = np.load('gnn/results1/mol_emb.npy')
    valid2 = np.load('gnn/results2/mol_emb.npy')
    print(valid1.shape)
    validation_set = pd.concat([pd.DataFrame(valid1), pd.DataFrame(valid2)],axis=1)
    torch.manual_seed(0)
    deepSVDD.build_network = build_network
    deepSVDD.build_autoencoder = build_autoencoder
    deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    net_name='gnn/model_100_1e-3_64_00005_gnn.pth'
    deep_SVDD.set_network(net_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    deep_SVDD.load_model('gnn/model_200_1e-3_32_0.0005_gnn.pth', True)
    scores = -1*ae_score(deep_SVDD, validation_set.iloc[:,:].values).cpu().detach().numpy()
    #pickle.dump(deep_SVDD.ae_net.get_attention_weights(), 'attention_weights')
    with open('attn_weights.pkl', 'wb') as f:
        pickle.dump(deep_SVDD.ae_net.get_attention_weights(), f)
    scaler= MinMaxScaler()
    scores_scaled = scaler.fit_transform(scores.reshape(-1,1)).ravel()
    dataset['scores'] = scores_scaled
    dataset_name = args.dataset_name.split('/')[-1].split('.')[0]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dataset.to_csv(f'{args.save_dir}/{dataset_name}.csv') 
    try:
        roc_auc = roc_auc_score(label.values, scores_scaled )
        print('total ROC-AUC', roc_auc)
        plots.ranking_plot(scores_scaled, label, names.values, args.save_dir) #_scaled
        plots.plot_confusion_matrix(label ,scores_scaled , args.threshold, args.save_dir)
    except: 
        pass#plots.ranking_plot(scores_scaled, label, names.values, args.save_dir) 

if __name__ == "__main__":     
    main()