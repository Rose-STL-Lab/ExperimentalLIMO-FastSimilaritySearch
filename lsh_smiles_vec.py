from utils import *
from models import *
import torch
from lshashpy3 import *
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
except NameError:
    raise Exception('No dm.pkl found, please run preprocess_data.py first')
vae.load_state_dict(torch.load('vae.pt'))
vae.eval()

smile = 'CC(N)(C)C1=CC=C(C(=O)N)C=C1N2CCCC(C3=CC(C(=O)C(C(C(=O)N)))=CC=C3)=C2'

def process_smiles(smiles_string: str):
    tensor_tuple = smile_to_vec(smiles_string, vae)  # Get tuple of tensors

    # Concatenate all tensors along dimension 1 (keeping as a row vector)
    concatenated_tensor = torch.cat(tensor_tuple, dim=1)

    # Flatten and convert to a list
    return concatenated_tensor.view(-1).cpu().detach().numpy().tolist()

df1 = pd.read_csv('SMILES_Big_Data_Set.csv')
df2 = pd.read_csv('real_data.csv')

vector_size = len(process_smiles(df2['LIMO compound'][0]))  # Determine vector size from a sample
lsh = LSHash(10, vector_size)

for ds in df2['Closest REAL compound'].dropna().tolist():
    try:
        vec = process_smiles(ds)
        lsh.index(vec, extra_data=ds)
    except:
        continue

results = []

# Query using SMILES from 'LIMO Compounds'
for query_smile in df2['LIMO compound'].dropna().tolist():
    try:
        query_vector = process_smiles(query_smile)
        top_n = 1
        nn = lsh.query(query_vector, num_results=top_n, distance_func="cosine")
        
        for ((vec, extra_data), distance) in nn:
            print(f"Query SMILE: {query_smile}")
            print(f"Closest Match: {extra_data}, Distance: {distance}\n")
            results.append({
                "Query SMILE": query_smile,
                "Closest Match": extra_data,
                "Distance": distance
            })
    except:
        results.append({
            "Query SMILE": query_smile,
            "Closest Match": "Could not process",
            "Distance": None
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('limo_matches.csv', index=False)
