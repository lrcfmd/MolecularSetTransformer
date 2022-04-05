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
from ccdc import io

from ccdc.io import MoleculeReader
csd_mol_reader = MoleculeReader('CSD')
mol = csd_mol_reader.molecule('CSATBR')
print(mol.smiles)


#cocrystals_data = pd.read_csv('C:/Users/kvriz/Desktop/ccdc_ml_cocrystals/1_extract_cocrystals/datasets/train_data/cocrystal_id.csv')
#cocrystals_data=cocrystals_data.dropna(axis=0)
#dataset_cocrystals = cocrystals_data[~cocrystals_data.name.str.contains("solvate")]
#dataset_cocrystals = dataset_cocrystals[~dataset_cocrystals.name.str.contains("clathrate")] 
#dataset_cocrystals.to_csv('datasets/train_data/all_cocrystals.csv',index=False)

#dataset_cocrystals = pd.read_csv('datasets/train_data/cocrystals2020_clean_functionality.csv')

#dataset_cocrystals.csd_id
#crystal_reader = io.CrystalReader('CSD') 
#[crystal_reader.crystal(i) for i in dataset_cocrystals.csd_id.values]
#for i in crystal_reader :#dataset_cocrystals.csd_id.values:
    #print(i)
    #crystal=crystal_reader.crystal(i)
    #print(crystal)
 #   with io.CrystalWriter("test/%s.cif"%i.identifier) as writer: 
  #          writer.write(i)

    #with io.CrystalWriter(os.path.join('test', '%s.cif'%i.identifier), append=True) as crystal_writer:
     #  crystal_writer.write(i)

#csd_reader = io.EntryReader('CSD')
#csd_reader.entry('ABEBUF').publication


#read the smiles of each id and calculate the MW .
# sort them based on MW
# print out those with exactly the same MW and drop them out

#year=[]
#for i in dataset_cocrystals.csd_id:
