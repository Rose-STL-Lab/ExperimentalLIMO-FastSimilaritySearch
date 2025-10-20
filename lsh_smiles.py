import bz2
import io
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import sys
import gc
from functools import partial
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from lshashpy3 import LSHash

# Configuration for 10B molecules
CHUNK_SIZE = 10_000_000  # Molecules per processing chunk
FINGERPRINT_SIZE = 1024  # Reduced size with folding
LSH_HASH_SIZE = 10
NUM_HASHTABLES = 15
MAX_PROCESSES = os.cpu_count() // 2  # Prevent memory overcommit

def read_smiles_chunked(filename):
    """Memory-mapped streaming reader with chunked processing"""
    cache_file = os.path.splitext(filename)[0] + ".mmap"
    dtype = np.dtype('S512')
    
    if os.path.exists(cache_file):
        print("Cache exists")
        return np.memmap(cache_file, dtype=dtype, mode='r')

        
    # Create memory-mapped file in two passes
    with bz2.open(filename, 'rb') as bz_file:
        with io.TextIOWrapper(bz_file, encoding='utf-8') as text_file:
            # First pass: count valid molecules
            valid_count = 0
            for i, line in enumerate(text_file):
                if valid_line(line):
                    valid_count += 1
                if i % 10_000_000 == 0:
                    print(f"Scanned {i:,} lines...")
            
            # Create memory-mapped array
            arr = np.memmap(cache_file, dtype=dtype, mode='w+', shape=(valid_count,))
            
            # Second pass: populate data
            bz_file.seek(0)
            text_file = io.TextIOWrapper(bz_file, encoding='utf-8')
            ptr = 0
            for i, line in enumerate(text_file):
                if valid_line(line):
                    arr[ptr] = line.split()[0].encode('utf-8')
                    ptr += 1
                if i % 10_000_000 == 0:
                    print(f"Processed {i:,} lines...")
                    
    print("Done Scanning and Processing")
    return np.memmap(cache_file, dtype=dtype, mode='r')

def valid_line(line):
    """Efficient line validation"""
    line = line.strip()
    return line and any(c in line for c in 'CNO[]()=#@+-')

def process_chunk(args):
    """Process a chunk of SMILES to LSH index"""
    chunk_idx, smiles_chunk, output_dir = args
    lsh = LSHash(LSH_HASH_SIZE, FINGERPRINT_SIZE, NUM_HASHTABLES)
    
    print(f"Processing chunk {chunk_idx} ({len(smiles_chunk):,} molecules)")
    fps = generate_fingerprints_bulk(smiles_chunk)
    
    for idx, fp in enumerate(fps):
        fp_array = np.zeros(FINGERPRINT_SIZE, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        lsh.index(fp_array, extra_data=smiles_chunk[idx])
        
    output_path = os.path.join(output_dir, f"index_{chunk_idx:05d}.lsh")
    lsh.store(output_path)
    return output_path

def generate_fingerprints_bulk(smiles_chunk):
    """Bulk fingerprint generation using RDKit optimizations"""
    mols = (Chem.MolFromSmiles(s) for s in smiles_chunk)
    fpg = AllChem.GetMorganFingerprintAsBitVect
    return [fpg(mol, 2, nBits=FINGERPRINT_SIZE) for mol in mols if mol]

def build_distributed_index(filename, output_dir, n_processes=MAX_PROCESSES):
    """Build distributed LSH index with memory control"""
    os.makedirs(output_dir, exist_ok=True)
    smiles_mmap = read_smiles_chunked(filename)

    print("Start processing in chunks")
    # Process in chunks using parallel workers
    with mp.Pool(processes=n_processes, maxtasksperchild=10) as pool:
        tasks = []
        for i in range(0, len(smiles_mmap), CHUNK_SIZE):
            print(f"Processing chunk {i//CHUNK_SIZE} ({i:,} molecules)")
            chunk = smiles_mmap[i:i+CHUNK_SIZE].tolist()
            tasks.append((i//CHUNK_SIZE, chunk, output_dir))
        
        # Process chunks with memory cleanup
        for result in pool.imap_unordered(process_chunk, tasks, chunksize=1):
            print(f"Completed {result}")
            gc.collect()
    
    return output_dir

def query_distributed_index(query_smiles, index_dir, batch_size=1000, n_processes=MAX_PROCESSES):
    """Parallel query processing with distributed indexes"""
    index_files = [os.path.join(index_dir, f) for f in os.listdir(index_dir) if f.endswith(".lsh")]
    
    with mp.Pool(processes=n_processes) as pool:
        results = []
        for query_batch in batched(query_smiles, batch_size):
            batch_results = pool.starmap(
                process_query_batch,
                [(query_batch, index_file) for index_file in index_files]
            )
            results.extend(aggregate_results(batch_results))
    
    return results

def process_query_batch(query_batch, index_file):
    """Process query batch against a single index file"""
    lsh = LSHash(LSH_HASH_SIZE, FINGERPRINT_SIZE, NUM_HASHTABLES)
    lsh.load(index_file)
    
    batch_results = []
    for smiles in query_batch:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            batch_results.append({"query": smiles, "matches": []})
            continue
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, FINGERPRINT_SIZE)
        fp_array = np.zeros(FINGERPRINT_SIZE)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        matches = lsh.query(fp_array, num_results=5, distance_func="hamming")
        processed = process_matches(smiles, matches)
        batch_results.append({"query": smiles, "matches": processed})
    
    return batch_results

def process_matches(query_smiles, matches):
    """Process and validate matches"""
    results = []
    for (vec, extra_data), distance in matches:
        try:
            similarity = DataStructs.TanimotoSimilarity(
                Chem.MolFromSmiles(query_smiles).GetMorganFingerprintAsBitVect(2, FINGERPRINT_SIZE),
                Chem.MolFromSmiles(extra_data).GetMorganFingerprintAsBitVect(2, FINGERPRINT_SIZE)
            )
            results.append({
                "smiles": extra_data,
                "similarity": similarity,
                "distance": distance
            })
        except:
            continue
    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:3]

# Helper functions
def batched(iterable, n):
    """Batch data into tuples of length n"""
    from itertools import islice
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def aggregate_results(result_chunks):
    """Combine results from parallel processing"""
    final = {}
    for chunk in result_chunks:
        for res in chunk:
            final.setdefault(res["query"], []).extend(res["matches"])
    return [{"query": q, "matches": sorted(ms, key=lambda x: x["similarity"], reverse=True)[:3]} 
            for q, ms in final.items()]

if __name__ == "__main__":
    # Build distributed index
    index_dir = build_distributed_index("../H27/MH27M000.cxsmiles.bz2", "lsh_indexes")
    
    # Process queries
    queries = pd.read_csv("queries.csv")["smiles"].dropna().tolist()
    results = query_distributed_index(queries, index_dir)
    
    # Save results
    pd.DataFrame(results).to_csv(
    "results.csv",
    index=False,          # Exclude index column
    header=True,          # Include column headers (default)
    encoding='utf-8',     # Standard encoding
    na_rep='NaN',         # Representation for missing values
    float_format='%.4f'   # Control floating point precision
    )
