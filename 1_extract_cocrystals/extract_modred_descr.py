import numpy as np
import pandas as pd
import os.path
import ccdc
from ccdc import search, io, molecule
from ccdc.io import MoleculeReader, CrystalReader, EntryReader
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors  #https://github.com/mordred-descriptor/mordred
calc = Calculator(descriptors, ignore_3D=True)

def Remove(duplicate):
    return list(set(duplicate))

def get_smiles_from_csd():
    ''' Read each CSD identifier and save the smiles  '''
    co_crystals = pd.read_csv('datasets/train_data/cocrystals2020.csv', encoding='latin1')
    co_crystals = co_crystals.iloc[:, :]
    #print(co_crystals.csd_id)
    smiles1=[]
    smiles2=[]
    year=[]
    for i in co_crystals.csd_id.values: 
        #print(i)
        csd = MoleculeReader('CSD')
        csd_reader = io.EntryReader('CSD')
        year.append(csd_reader.entry(i).publication.year)
        mol = csd.molecule(i)
        smi= mol.smiles
        smi = smi.split('.')
        smi=Remove(smi)
        smiles1.append(smi[0])
        smiles2.append(smi[1])
        #print(len(smiles1))
    #cocrystal_data = pd.concat([co_crystals , pd.DataFrame(smiles1, columns=['smiles1']), pd.DataFrame(smiles2, columns=['smiles2']),
    #pd.DataFrame(year, columns=['year'])], axis=1)
    #cocrystal_data.to_csv('datasets/train_data/all_cocrystals_info.csv')
    return co_crystals, smiles1, smiles2
 
def modred_descriptors():
    ''' Calculate the descriptors after converting the smiles to the canonical form '''
    
    descriptors_mol1 =[]
    #co_crystals, smiles1, smiles2 = get_smiles_from_csd()
    val_data = pd.read_csv('datasets/validation_data/all_validation_sets.csv', encoding='latin1')
    smiles1 = val_data.smiles1
    for mol in smiles1[:]:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
            descriptors_mol1.append(calc(Chem.MolFromSmiles(smi)))
        except TypeError:
            descriptors_mol1.append('none')
    dataset1 = pd.DataFrame(descriptors_mol1,columns=calc.descriptors[:])
    #dataset1.to_csv('training_set/modred/training_API_modred.csv', index=False)
    dataset1 = pd.DataFrame(dataset1.values, columns=calc.descriptors[:])
    #pd.concat([co_crystals.csd_id, pd.DataFrame(smiles1, columns=['smiles']),
    #dataset1],axis=1).to_csv('datasets/train_data/dataset1_mordred.csv', index=False)
    pd.concat([val_data.Dataset, pd.DataFrame(smiles1, columns=['smiles']),
     dataset1],axis=1).to_csv('datasets/validation_data/val_set1_mordred.csv', index=False)

    # Calculate the descriptors for the second dataset (first co-former)
    #dataset_mol2 = pd.read_csv('dataset_mol2.csv')
    descriptors_mol2 =[]
    #print(smiles2)
    smiles2 = val_data.smiles2
    for mol in smiles2:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
            descriptors_mol2.append(calc(Chem.MolFromSmiles(smi)))
        except:
            descriptors_mol2.append('none')
    # Save dataset2 with the descriptors
    dataset2 = pd.DataFrame(descriptors_mol2, columns=calc.descriptors[:])
    dataset2 = pd.DataFrame(dataset2.values, columns=calc.descriptors[:])
    #pd.concat([co_crystals.csd_id, pd.DataFrame(smiles2, columns=['smiles']),
     #dataset2],axis=1).to_csv('datasets/train_data/dataset2_mordred.csv', index=False)
    pd.concat([val_data.Dataset, pd.DataFrame(smiles2, columns=['smiles']),
     dataset2],axis=1).to_csv('datasets/validation_data/val_set2_mordred.csv', index=False)

def main():
    #get_smiles_from_csd()
    modred_descriptors()

if __name__ == "__main__":     
    main()
