import numpy as np
import pandas as pd
import os.path
import ccdc
from ccdc import search, io, molecule
from ccdc.io import MoleculeReader, CrystalReader, EntryReader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from ccdc import conformer
from ccdc.molecule import Molecule
import pubchempy as pcp

out = ['CONYAF10', 'ZAYQEV', 'MIPWEN', 'NELQAX', 'IWOLIQ','ULOCII', 'NENPAA',
        'ZOYPIP']
        
def Remove(duplicate):
    return list(set(duplicate))

#def get_molecule_file():
#    co_crystals = pd.read_csv('data/cocrystals2020.csv', encoding='latin1')

def get_3d(smile):
    ''' Generates the 3D configuration, using CCDC conformer generator,
    of a molecules given its SMILES string. 
    Requires ccdc python API 2021.1 '''

    conformer_generator = conformer.ConformerGenerator()
    conformer_generator.settings.max_conformers = 1
    mol = Molecule.from_string(smile)
    conformers = conformer_generator.generate(mol)
    return conformers[0]

def feature_first(a, b, w_a, w_b):
    ''' Sorting the smiles strings by having the heavier first '''
    f, s = (a, b) if w_a > w_b else (b,a)
    return f,s

def feature_apply(smi):
    return feature_first(smi[0], smi[1], Chem.Descriptors.MolWt(Chem.MolFromSmiles(smi[0])), 
    Chem.Descriptors.MolWt(Chem.MolFromSmiles(smi[1])))

def stereo_smiles_from_csd():
    ''' Read each CSD identifier and save the smiles  '''
    co_crystals = pd.read_csv('data/csd_cocrystals2020.csv', encoding='latin1')
    co_crystals = co_crystals.iloc[:, :]
    co_crystals = co_crystals[~co_crystals.csd_id.isin(out)]
    #co_crystals = pd.DataFrame(co_crystals.values, columns=co_crystals.columns.values)
    #print(co_crystals.csd_id)
    smiles1=[]
    smiles2=[]
    
    for mol in co_crystals.csd_id.values[:]: 
        csd = MoleculeReader('CSD')
        csd_reader = io.EntryReader('CSD')
        mol = csd.molecule(mol)
        smi= mol.to_string('smiles')
        smi = smi.split('.')
        smi=Remove(smi)
        
    cocrystal_data = pd.concat([ pd.DataFrame(co_crystals.csd_id, columns=['csd_id']),
      pd.DataFrame(smiles1, columns=['smiles1']), pd.DataFrame(smiles2, columns=['smiles2'])],axis=1)
    cocrystal_data.to_csv('csd_cocrystals_stereosmiles.csv')
        
def main():
    stereo_smiles_from_csd()

if __name__ == "__main__":     
    main()
