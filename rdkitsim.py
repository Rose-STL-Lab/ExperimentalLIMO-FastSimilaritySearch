from utils import *
from models import *
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pandas as pd

df = pd.read_csv('SMILES.csv')
c_smiles = []
for ds in df[' SMILES']:
    try:
        cs = Chem.CanonSmiles(ds)
        c_smiles.append(cs)
    except:
        print('Invalid SMILES:', ds)

ms = [Chem.MolFromSmiles(x) for x in c_smiles]

print(ms)
print('\n')

fps = [FingerprintMols.FingerprintMol(x, minPath=1, maxPath=7, fpSize=2048,
                               bitsPerHash=2, useHs=True, tgtDensity=0.0,
                               minSize=128) for x in ms]

print(fps[0])
print('\n')
print(fps[1:])
print('\n')

# print(fps)
qu, ta, sim = [], [], []
for n in range(len(fps)-1): # -1 so the last fp will not be used
    s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n+1:])
    print(c_smiles[n], c_smiles[n+1:]) # witch mol is compared with what group
    # collect the SMILES and values
    for m in range(len(s)):
        qu.append(c_smiles[n])
        ta.append(c_smiles[n+1:][m])
        sim.append(s[m])

d = {'query':qu, 'target':ta, 'Similarity':sim}
df_final = pd.DataFrame(data=d)
df_final = df_final.sort_values('Similarity', ascending=False)
print(df_final)

df_final.to_csv('result.csv', index=False, sep=',')
