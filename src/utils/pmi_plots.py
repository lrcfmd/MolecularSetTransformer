import random
import argparse
import random
from warnings import filterwarnings
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import rdkit
from rdkit import Chem, RDLogger
import glob
from rdkit.Chem import AllChem, Descriptors3D
import seaborn as sns
from matplotlib import rc
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle

plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["axes.labelsize"] = "xx-large"
plt.rcParams["axes.labelweight"]= "normal"
plt.rcParams["xtick.labelsize"] = "xx-large"
plt.rcParams["ytick.labelsize"] = "xx-large"
plt.rcParams.update({
    "text.usetex": False,
    #"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def get_normalized_principle_moment_ratios():
    #molecules = zinc_smiles.smiles
    #m1=[]
    #for file in sorted(glob.glob("/content/drive/MyDrive/zinc15_new/mol_files/add_hydrogen/*.mol")):

    #name = (file.split('.')[0]).split('/')[-1]
  
     # m = Chem.MolFromMolFile(file)
    #m1.append(m)
    # shuffle the molecules before plotting
    molecules = [Chem.MolFromMolFile(mol) for mol in sorted(glob.glob("../../data/coformer1/*.mol"))]# sorted(glob.glob("/content/drive/MyDrive/zinc15_new/mol_files/add_hydrogen/*.mol")) ] #[mol for mol in m1]
    #name = [(file.split('.')[0]).split('/')[-1] for file in sorted(glob.glob("/content/zinc20_updated/*"))]
    #print(name)
    #if name == 'ZINC000085548520':
     # print(zinc20_lumo_dict['ZINC000085548520'])

    print(len(molecules))
    #random.shuffle(molecules)

    # create a list of all the NPRs
    npr1 = list()
    npr2 = list()
    fails = 0
    n_mols = 0
    for mol in molecules:
        try:
            #mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            #AllChem.EmbedMolecule(mol)  # generate a 3D embedding
            npr1.append(rdkit.Chem.Descriptors3D.NPR1(mol))
            npr2.append(rdkit.Chem.Descriptors3D.NPR2(mol))
            n_mols += 1
            #print(npr2)
        except:
            fails += 1
            print(mol)
        if n_mols == 10000:
            print("-- Truncating at 10K")
            break
    
    print(len(npr1))
    print(len(npr2))
    nbins = 30
    k = kde.gaussian_kde((npr1, npr2))
    xi, yi = np.mgrid[0:1:nbins*1j, 0.5:1:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # plot the NRP on a 2D map
    fig = plt.figure(figsize =(10, 8))
    fig.patch.set_facecolor('white')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.grid(False)

    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.right'] = True
    #c=[zinc20_homo_dict[i] for i in name]
    #print(max(c))
    #print(min(c))
    #facecolors = [cm.viridis(x) for x in c]
    plt.hexbin(npr1, npr2)#, gridsize=nbins, C=zi,   cmap=plt.cm.jet_r, mincnt=1, extent=(0, 1, 0.5, 1), alpha=0.8, zorder=6)#, vmin=0, vmax=150, zorder=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15) 
    #cbar.set_label('kernel density', fontsize=16)
    #cbar.set_label('LUMO$_{+1}$-LUMO$_{+2}$ degeneracy', fontsize=16)
    cbar.set_label('HOMO-LUMO degeneracy', fontsize=16)

    #plt.contour(xi, yi, zi.reshape(xi.shape), levels=5, zorder=1)
    plt.fill([0, 0, 0.5], [0.5, 1, 0.5], "white", zorder=2)  # `white out' the bottom left corner of the plot
    plt.fill([1, 1, 0.5], [0.5, 1, 0.5], "white", zorder=3)  # `white out' the bottom right corner of the plot
    plt.plot([0, 0.5], [1, 0.5], color="lightsteelblue", linewidth=3.5, zorder=4)
    plt.plot([0.5, 1], [0.5, 1], color="lightsteelblue", linewidth=3.5, zorder=5)
    plt.plot([0, 1], [1, 1], color="lightsteelblue", linewidth=3.5, zorder=0)
    #plt.axvline(x=3.5, alpha=0.5)
    plt.ylabel("NPR2", fontsize=16)
    plt.xlabel("NPR1", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    matplotlib.rc('axes',edgecolor='black')
    #ax.spines['bottom'].set_color('black')
    #ax.spines['top'].set_color('black') 
    #ax.spines['right'].set_color('black')
    #ax.spines['left'].set_color('black')
    #plt.plot(loss.Epochs.values, loss.Column7.values, '-o')
    plt.ylim(0.4, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.savefig("../../data/figures/npr_mol1.png", dpi=600,  bbox_inches='tight')
    #print("-- File saved in ", smi_file[:-4] + "_npr.png")

    # return the values
    return npr1, npr2, fails


get_normalized_principle_moment_ratios()
