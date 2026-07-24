# faiss_smiles_fingerprint_mp.py

FAISS-based molecular similarity search over large compound libraries (tested at 849M compounds) using binary Morgan fingerprints. Designed to run within 16 GiB of RAM via sharded index building and memory-aware batch processing.

## How it works

1. **Fingerprint generation** — Each SMILES string is converted to a 1024-bit ECFP4 (Morgan radius 2) fingerprint packed into 128 bytes. This runs in parallel across worker processes using `multiprocessing.Pool`.
2. **Sharded index building** — Fingerprints are accumulated into shards of up to 30M compounds (~7.7 GB each). Each shard is trained as an IVF binary index (`IndexBinaryIVF`) and written to disk before the next shard begins, keeping peak RAM usage below ~15 GB.
3. **SQLite mapping** — All SMILES strings are stored in a SQLite database keyed by their global FAISS index position, allowing hit retrieval after search.
4. **Manifest file** — A JSON manifest tracks shard paths, compound counts, and input-line offsets so interrupted builds can be resumed exactly.

Similarity is measured by **Hamming distance** on packed binary fingerprints. Lower distance = more similar.

## Dependencies

- Python 3.9+
- `rdkit`
- `faiss-cpu` (or `faiss-gpu`)
- `numpy`

## Usage

### Build index

```bash
python faiss_smiles_fingerprint_mp.py build \
    --input 2025.02_Enamine_REAL_HAC_24_849M_CXSMILES.cxsmiles.bz2 \
    --output-prefix enamine_849m \
    --smiles-db enamine_849m_smiles.db
```

### Resume an interrupted build

```bash
python faiss_smiles_fingerprint_mp.py build \
    --input data.cxsmiles.bz2 \
    --output-prefix enamine_849m \
    --smiles-db enamine_849m_smiles.db \
    --start-shard 5
```

### Query

```bash
python faiss_smiles_fingerprint_mp.py query \
    --manifest enamine_849m_manifest.json \
    --smiles-db enamine_849m_smiles.db \
    --smiles "CCO" \
    --k 10
```

Returns the top-k nearest neighbors by Hamming distance, loading one shard at a time to stay within memory limits.

### Test run with a compound limit

```bash
python faiss_smiles_fingerprint_mp.py build \
    --input data.cxsmiles.bz2 \
    --output-prefix test \
    --smiles-db test_smiles.db \
    --max-compounds 1000000
```

## CLI reference

| Flag | Command | Description |
|------|---------|-------------|
| `--input / -i` | build | bz2-compressed CXSMILES input file |
| `--output-prefix / -o` | build | Prefix for shard `.faiss` files (default: `enamine`) |
| `--smiles-db / -s` | build/query | Path to SQLite SMILES database |
| `--max-compounds / -m` | build | Cap on compounds processed (for testing) |
| `--start-shard` | build | Shard number to resume from after a crash |
| `--num-workers / -w` | build | Worker processes for fingerprint generation (default: CPU count − 1) |
| `--manifest / -m` | query | Path to manifest JSON |
| `--smiles / -q` | query | Query SMILES string |
| `--k / -k` | query | Number of nearest neighbors to return (default: 10) |

## Python API

```python
from faiss_smiles_fingerprint_mp import (
    search_sharded_lazy,          # single query, one shard in RAM at a time
    search_sharded_lazy_batch,    # batch of queries, one shard in RAM at a time
    load_sharded_index,           # load all shards into RAM (fast, high memory)
    search_sharded,               # query pre-loaded shards
)

# Single query (memory-efficient)
results = search_sharded_lazy(
    manifest_path="enamine_manifest.json",
    smiles_db_path="enamine_smiles.db",
    query_smiles="c1ccccc1",
    k=10,
)
# returns [(smiles, hamming_distance), ...]

# Batch query (amortises shard load cost across all queries)
results_list = search_sharded_lazy_batch(
    manifest_path="enamine_manifest.json",
    smiles_db_path="enamine_smiles.db",
    query_smiles_list=["c1ccccc1", "CCO", "CC(=O)O"],
    k=10,
)
# returns one result list per input query
```

## Memory profile

| Phase | Peak RAM |
|-------|----------|
| Build (per shard, 30M compounds) | ~15 GB |
| Query — lazy (one shard at a time) | ~8 GB per shard |
| Query — all shards loaded | ~8 GB × number of shards |

Worker processes are fully shut down before each FAISS build phase to reclaim their RSS (each worker holds ~500 MB–1 GB for RDKit framework initialisation on macOS).

## Key constants (tunable at top of file)

| Constant | Default | Purpose |
|----------|---------|---------|
| `FINGERPRINT_SIZE` | 1024 | Fingerprint bit length |
| `SHARD_SIZE` | 30,000,000 | Compounds per shard |
| `NLIST_PER_SHARD` | 2048 | IVF cluster count per shard |
| `NPROBE` | 64 | Clusters searched at query time (recall vs. speed) |
| `TRAINING_SAMPLE_SIZE` | 500,000 | Samples used to train each IVF index |
| `NUM_WORKERS` | CPU count − 1 | Parallel fingerprint workers |
| `WORKER_CHUNK_SIZE` | 2000 | SMILES per worker task |

Increasing `NPROBE` improves recall at the cost of query latency. Increasing `SHARD_SIZE` reduces the number of shards (faster queries) but raises peak build-time RAM.

## Input format

Accepts bz2-compressed CXSMILES files where the SMILES string is the first whitespace-separated field on each line. A header line containing `smiles` (case-insensitive) is skipped automatically.
