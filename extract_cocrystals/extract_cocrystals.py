# Import the libraries
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import os.path
import ccdc
from ccdc import search, io, molecule
from ccdc.io import MoleculeReader, CrystalReader, EntryReader
import csv
from collections import Counter
from itertools import groupby
import argparse
import time
import clean_smiles

def remove_polymorphs(lst):
    ''' 
    Checking if the first 6 letters of the ccdc id are the same 
    '''
    res = []
    for g, l in groupby(sorted(lst), lambda x: x[:6]):
        res.append(next(l))
    return res

def Remove(duplicate):
    return list(set(duplicate))

def search_cocrystals(filter_solvents=True):
    '''
    Search the whole CSD for structures that contain two different molecules
    with the specific settings
    '''
    start_time = time.clock()
    csd = MoleculeReader('CSD')
    entry_reader = EntryReader('CSD')
    settings = search.Search.Settings()
    settings.only_organic = True
    settings.not_polymeric = True
    settings.has_3d_coordinates = True
    settings.no_disorder = True
    settings.no_errors = True
    settings.no_ions = True
    settings.no_metals = True
    pairs=[]
    for entry in csd:
        #if len(pairs)==100:
        #    break
        if settings.test(entry):
            mol = csd.molecule(entry.identifier)
            mol.normalise_labels()
            smi= mol.smiles
            if smi !=  None:
                smi = smi.split('.')
                # We make sure that the structure consist of two different molecules
                if len(Remove(smi)) == 2:                
                    pairs.append(mol.identifier)            
    # clean the list from solvents
    if filter_solvents:
        print('Solvates and hydrates will be removed')
        solvates=[]
        name_dict={}
        for mol1 in pairs:
            mol = csd.molecule(mol1)
            e=entry_reader.entry(mol1)
            name_dict[mol1]=e.chemical_name
            for i in range(0, (len(mol.components))):
                if mol.components[i].smiles in clean_smiles.SOLVENT_SMILES:
                    solvates.append(mol.identifier)    
        solvates = Remove(solvates)
        final_cocrystals = [x for x in pairs if x not in solvates]   
        #print(name_dict) 
    else:
        final_cocrystals=pairs
    # Clean the list from polymorphs
    cocrystals = remove_polymorphs(final_cocrystals)
    #print the time
    end_time = time.clock()
    name=[]
    name= [name_dict[i] for i in cocrystals]
    cocrystals_data= pd.concat([pd.DataFrame(cocrystals, columns=['csd_id']), pd.DataFrame(name, columns=['name'])], axis=1)
    cocrystals_data=cocrystals_data.dropna(axis=0)
    dataset_cocrystals = cocrystals_data[~cocrystals_data.name.str.contains("solvate")]
    dataset_cocrystals = dataset_cocrystals[~dataset_cocrystals.name.str.contains("clathrate")] 
     
    print(end_time-start_time)
    dataset_cocrystals.to_csv('datasets/train_data/all_cocrystals.csv',index=False)
    return cocrystals

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--solvent', default=True, action='store_true', help='Remove solvents or not')
    args = parser.parse_args()
    cocrystals = search_cocrystals(args.solvent)

if __name__ == "__main__":
    main()
