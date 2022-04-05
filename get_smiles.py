import pubchempy as pcp
import pandas as pd

data=pd.read_csv('csd_cocrystals_stereosmiles.csv')
names2=[]

for name in data.smiles2:
	name2=[]
	try:
		for compound in pcp.get_compounds(name, 'smiles'):
			#print(compound.isomeric_smiles)
			name2.append(compound.isomeric_smiles)
		try:
			names2.append(name2[0])
		except:
			names2.append(0)
	except:
		names2.append(name)

data['Smiles2'] = pd.DataFrame(names2)
data.to_csv('csd_cocrystals_stereosmiles.csv') 