"""
FAISS-based molecular similarity search using Morgan fingerprints.
Optimized for large-scale datasets (849M+ compounds) with limited memory (16 GiB).

Uses sharded index building to stay within memory constraints:
- 849M compounds × 256 bytes = 217 GB total fingerprint data
- With 16 GiB RAM, we build ~17 shards of ~50M compounds each
- Each shard uses ~13 GB RAM during build (50M × 256 bytes + overhead)
- At query time, search all shards and merge results

Output files:
- enamine_shard_00.faiss, enamine_shard_01.faiss, ... (FAISS index shards)
- enamine_smiles.db (SQLite database mapping FAISS IDs to SMILES)
- enamine_manifest.txt (list of shard files for querying)
"""
import bz2
import gc
import json
import os
import argparse
import sqlite3
import tempfile
import multiprocessing
from pathlib import Path
from typing import Optional, Iterator, Tuple, List

import numpy as np
# rdkit and faiss are imported lazily (inside the functions that use them) so
# that spawned worker processes start up with a clean module and only import
# what they need at task-execution time.  On macOS, importing either library
# in a freshly-spawned subprocess can trigger framework initialisation that
# hangs or deadlocks before any task is submitted.

# Configuration
FINGERPRINT_SIZE = 1024  # Number of bits in fingerprint
FINGERPRINT_BYTES = FINGERPRINT_SIZE // 8  # 256 bytes per fingerprint

# Memory-aware batch sizes (tuned for 16 GiB RAM)
# Each fingerprint = 256 bytes
# Shard size: 50M compounds = 12.8 GB fingerprints + ~2GB overhead = ~15GB peak
SHARD_SIZE = 30_000_000  # 50M compounds per shard (~13 GB RAM usage)
TRAINING_SAMPLE_SIZE = 500_000  # 500K samples for IVF training per shard (~128 MB)
BATCH_SIZE = 1_000_000  # Process 1M compounds at a time for DB writes
PROGRESS_INTERVAL = 1_000_000  # Print progress every 1M compounds

# FAISS IVF parameters (per-shard, scaled for 30M compounds)
# sqrt(30M) ≈ 5.5K; 2048 clusters balances build speed vs recall
NLIST_PER_SHARD = 2048  # Number of clusters for IVF index per shard
NPROBE = 64  # Number of clusters to search at query time
ADD_BATCH_SIZE = 1_000_000  # Vectors per index.add() call (cache efficiency)

# Multiprocessing configuration
# Default to leaving one core free for the main process / OS
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
WORKER_CHUNK_SIZE = 2000  # SMILES per chunk sent to each worker


def smiles_to_binary_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to packed binary fingerprint for FAISS binary index."""
    # Lazy imports: keeps worker processes lightweight at spawn time.
    # Python caches imports in sys.modules, so only the first call per process
    # pays the import cost; subsequent calls are effectively free.
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate Morgan fingerprint (ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_SIZE)

        # Convert to packed uint8 array for FAISS binary index
        fp_array = np.zeros(FINGERPRINT_SIZE, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, fp_array)

        # Pack bits into bytes (FAISS expects uint8 with 8 bits per byte)
        packed = np.packbits(fp_array)
        return packed
    except Exception:
        return None


def _process_smiles_chunk(chunk: List[Tuple[int, str]]) -> List[Tuple[int, str, np.ndarray]]:
    """
    Worker function: convert a chunk of (temp_key, smiles) pairs to fingerprints.

    Runs in a subprocess. Returns (temp_key, smiles, fingerprint) triples for
    successful conversions only. Returning the smiles avoids the need for a
    shared pending dict in the main process and the threading issues that come
    with it (imap_unordered consumes the input iterator on a background thread).
    """
    results = []
    for temp_key, smiles in chunk:
        fp = smiles_to_binary_fingerprint(smiles)
        if fp is not None:
            results.append((temp_key, smiles, fp))
    return results


def stream_smiles_from_bz2(filepath: str) -> Iterator[Tuple[int, str]]:
    """
    Stream SMILES strings from a bz2-compressed CXSMILES file.

    Yields (line_index, smiles) tuples. Memory efficient - only one line at a time.
    """
    with bz2.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # CXSMILES format: SMILES is first column (tab or space separated)
            # Skip header if present
            if idx == 0 and ('smiles' in line.lower() or 'SMILES' in line):
                continue
            # Extract SMILES (first whitespace-separated field)
            parts = line.split()
            if parts:
                yield idx, parts[0]


def create_smiles_database(db_path: str) -> sqlite3.Connection:
    """Create SQLite database for storing SMILES strings with their indices."""
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')  # Better concurrent performance
    conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
    conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
    conn.execute('''
        CREATE TABLE IF NOT EXISTS smiles (
            idx INTEGER PRIMARY KEY,
            smiles TEXT NOT NULL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_smiles ON smiles(idx)')
    return conn


def build_index(
    input_file: str,
    output_prefix: str,
    smiles_db_output: str,
    max_compounds: Optional[int] = None,
    start_shard: int = 0,
    num_workers: int = NUM_WORKERS
):
    """
    Build sharded FAISS index from bz2-compressed SMILES file.

    Memory-efficient approach using sharding:
    1. Stream through file, building one shard at a time
    2. Each shard contains up to SHARD_SIZE compounds (~50M)
    3. Train IVF on first batch of each shard, then add remaining
    4. Save each shard to disk before starting next one
    5. Store all SMILES in SQLite database

    Fingerprint generation is parallelised across `num_workers` processes
    using multiprocessing.Pool.imap_unordered so that CPU-bound RDKit work
    runs on multiple cores while the main process handles I/O and FAISS.

    Args:
        input_file: Path to bz2-compressed CXSMILES file
        output_prefix: Prefix for output files (e.g., "enamine" -> enamine_shard_00.faiss)
        smiles_db_output: Path for SQLite database
        max_compounds: Maximum compounds to process (for testing)
        start_shard: Resume from this shard number (for crash recovery)
        num_workers: Number of worker processes for fingerprint generation
    """
    print(f"Building sharded FAISS index from: {input_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"SMILES database: {smiles_db_output}")
    print(f"Memory budget: ~16 GiB")
    print(f"Shard size: {SHARD_SIZE:,} compounds (~{SHARD_SIZE * FINGERPRINT_BYTES / 1e9:.1f} GB)")
    print(f"IVF clusters per shard: {NLIST_PER_SHARD:,}")
    print(f"Worker processes: {num_workers} (chunk size: {WORKER_CHUNK_SIZE:,})")
    print()

    # Ensure output directories exist (for both the shard files and the DB)
    for path in (smiles_db_output, f"{output_prefix}_shard_00.faiss"):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # Create/open SMILES database
    if start_shard == 0 and os.path.exists(smiles_db_output):
        os.remove(smiles_db_output)

    conn = create_smiles_database(smiles_db_output)
    cursor = conn.cursor()

    # Track shard files
    shard_files = []
    line_offsets = []   # input-line offset at which each shard ended (for resume)
    manifest_path = f"{output_prefix}_manifest.json"

    # Current shard state
    current_shard = 0
    # Pre-allocate a reusable fingerprint matrix for the shard.
    # Avoids building a Python list of 30M small arrays and calling np.vstack.
    shard_fps_array = np.empty((SHARD_SIZE, FINGERPRINT_BYTES), dtype=np.uint8)
    # On resume each completed shard holds exactly SHARD_SIZE valid compounds,
    # so the correct starting global_idx is start_shard * SHARD_SIZE.
    global_idx = start_shard * SHARD_SIZE
    total_processed = 0
    skip_until_idx = 0

    # If resuming, calculate where to start
    if start_shard > 0:
        current_shard = start_shard
        # Load existing manifest to recover the exact input-line offset where
        # each completed shard ended.  This avoids the unit mismatch between
        # SHARD_SIZE (valid fingerprints) and the raw line count in the file,
        # which would otherwise cause the first few thousand already-indexed
        # compounds to be silently re-processed into the new shard.
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                shard_files = manifest.get('shards', [])[:start_shard]
                line_offsets = manifest.get('line_offsets', [])[:start_shard]
        if len(line_offsets) >= start_shard:
            skip_until_idx = line_offsets[start_shard - 1]
        else:
            # Manifest predates line_offsets tracking; fall back to the
            # approximate value and warn the user.
            skip_until_idx = start_shard * SHARD_SIZE
            print(f"  Warning: manifest has no line_offsets; using approximate "
                  f"skip of {skip_until_idx:,} lines.  A small number of "
                  f"compounds near shard boundaries may be re-indexed.")
        print(f"Resuming from shard {start_shard}, skipping first {skip_until_idx:,} input lines")

    print("=" * 60)
    print("Processing compounds and building shards...")
    print("=" * 60)

    # A fresh worker pool is opened for each shard's fingerprint phase and is
    # fully closed before _build_single_shard is called.  On macOS each worker
    # holds 500 MB–1 GB RSS (RDKit framework initialisation).  With the old
    # single-pool design those workers were idle but resident throughout the
    # 10–30 min FAISS build, pushing total RSS past available RAM and causing
    # the OS to kill the process.  Closing the pool first frees that memory
    # before peak FAISS usage.  maxtasksperchild=500 also caps per-worker RSS
    # growth from processing millions of SMILES over a long run.
    stream = stream_smiles_from_bz2(input_file)
    stream_exhausted = False
    # Compounds returned by the pool after SHARD_SIZE was reached; they seed
    # the next shard so no fingerprint work is lost.
    carry: List[Tuple[int, str, np.ndarray]] = []
    last_progress = 0

    while not stream_exhausted or carry:
        # ── Phase 1: fingerprint collection (pool open) ───────────────────────
        shard_fp_count = 0
        shard_smiles = []
        last_shard_temp_key = 0
        new_carry: List[Tuple[int, str, np.ndarray]] = []

        # Seed shard buffer with overflow compounds from the previous pool pass.
        for temp_key, smiles, fp in carry:
            shard_fps_array[shard_fp_count] = fp
            shard_fp_count += 1
            shard_smiles.append((global_idx, smiles))
            global_idx += 1
            last_shard_temp_key = temp_key
        carry = []

        if not stream_exhausted:
            # Emit enough input lines to fill the remainder of this shard.
            # Invalid SMILES are rare in Enamine REAL (<1 %), so no large
            # buffer is needed; any overflow goes into new_carry for the next
            # shard iteration.
            lines_to_emit = max(0, SHARD_SIZE - shard_fp_count) + WORKER_CHUNK_SIZE

            def _gen_chunks():
                nonlocal total_processed, stream_exhausted
                chunk = []
                emitted = 0
                for _, smiles_str in stream:
                    total_processed += 1

                    if max_compounds and total_processed > max_compounds:
                        stream_exhausted = True
                        break

                    # Skip input lines when resuming from a previous run.
                    if total_processed <= skip_until_idx:
                        if total_processed % PROGRESS_INTERVAL == 0:
                            print(f"  Skipping... {total_processed:,} / {skip_until_idx:,}")
                        continue

                    chunk.append((total_processed, smiles_str))
                    emitted += 1

                    if len(chunk) >= WORKER_CHUNK_SIZE:
                        yield chunk
                        chunk = []

                    if emitted >= lines_to_emit:
                        break
                else:
                    stream_exhausted = True

                if chunk:
                    yield chunk

            effective_workers = min(num_workers, max(1, lines_to_emit // WORKER_CHUNK_SIZE))
            print(f"  Fingerprinting shard {current_shard} "
                  f"({effective_workers} worker(s))...")
            with multiprocessing.Pool(
                processes=effective_workers, maxtasksperchild=500
            ) as pool:
                for batch_results in pool.imap_unordered(
                    _process_smiles_chunk, _gen_chunks()
                ):
                    batch_results.sort(key=lambda x: x[0])

                    for temp_key, smiles, fp in batch_results:
                        if shard_fp_count < SHARD_SIZE:
                            shard_fps_array[shard_fp_count] = fp
                            shard_fp_count += 1
                            shard_smiles.append((global_idx, smiles))
                            global_idx += 1
                            last_shard_temp_key = temp_key
                        else:
                            new_carry.append((temp_key, smiles, fp))

                    if global_idx - last_progress >= PROGRESS_INTERVAL:
                        last_progress = global_idx
                        print(f"  Processed {total_processed:,} lines, "
                              f"{global_idx:,} valid compounds, "
                              f"shard {current_shard} has {shard_fp_count:,}")
            # Pool closed here — all worker processes have exited and their
            # RSS is released before the FAISS build below.

        carry = new_carry

        if shard_fp_count == 0:
            break

        # ── Phase 2: FAISS index build (no worker processes running) ─────────
        shard_path = f"{output_prefix}_shard_{current_shard:02d}.faiss"
        _build_single_shard(shard_fps_array[:shard_fp_count], shard_path, current_shard)
        shard_files.append(shard_path)
        line_offsets.append(last_shard_temp_key)

        print(f"  Saving {len(shard_smiles):,} SMILES to database...")
        cursor.executemany('INSERT INTO smiles VALUES (?, ?)', shard_smiles)
        conn.commit()

        _save_manifest(manifest_path, shard_files, global_idx, line_offsets)

        print(f"  Shard {current_shard} complete. Total indexed: {global_idx:,}")
        print()
        current_shard += 1
        gc.collect()

    conn.close()

    # Manifest is saved after every shard; write it one final time to make
    # total_compounds authoritative for the completed run.
    _save_manifest(manifest_path, shard_files, global_idx, line_offsets)

    # Print summary
    print()
    print("=" * 60)
    print("Build complete!")
    print("=" * 60)

    total_index_size = sum(os.path.getsize(f) for f in shard_files)
    db_size = os.path.getsize(smiles_db_output)

    print(f"  Total compounds indexed: {global_idx:,}")
    print(f"  Number of shards: {len(shard_files)}")
    print(f"  Total index size: {total_index_size / 1e9:.2f} GB")
    print(f"  SMILES database size: {db_size / 1e9:.2f} GB")
    print(f"  Manifest file: {manifest_path}")
    print()
    print("Shard files:")
    for sf in shard_files:
        print(f"  - {sf} ({os.path.getsize(sf) / 1e9:.2f} GB)")


def _build_single_shard(
    fps_array: np.ndarray,
    output_path: str,
    shard_num: int
):
    """Build a single FAISS shard from a pre-allocated uint8 fingerprint matrix."""
    import faiss  # Lazy import: keep FAISS out of worker processes

    # Ensure the output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Building shard {shard_num} with {len(fps_array):,} compounds...")

    # fps_array is already a pre-allocated uint8 matrix — no vstack needed.
    n_samples = len(fps_array)
    train_size = min(TRAINING_SAMPLE_SIZE, n_samples)

    if n_samples > train_size:
        train_indices = np.random.choice(n_samples, train_size, replace=False)
        train_array = fps_array[train_indices]
    else:
        train_array = fps_array

    # Determine number of clusters
    actual_nlist = min(NLIST_PER_SHARD, n_samples // 40)
    actual_nlist = max(actual_nlist, 100)

    print(f"    Training IVF with {actual_nlist:,} clusters on {len(train_array):,} samples...")

    # Create and train index
    quantizer = faiss.IndexBinaryFlat(FINGERPRINT_SIZE)
    index = faiss.IndexBinaryIVF(quantizer, FINGERPRINT_SIZE, actual_nlist)
    index.train(train_array)

    del train_array
    gc.collect()

    # Add fingerprints in batches for better cache efficiency.
    # Each batch triggers one IVF quantization pass over ADD_BATCH_SIZE vectors.
    print(f"    Adding {len(fps_array):,} fingerprints to index (batch size {ADD_BATCH_SIZE:,})...")
    for start in range(0, n_samples, ADD_BATCH_SIZE):
        index.add(fps_array[start:start + ADD_BATCH_SIZE])

    # Save to disk
    print(f"    Saving to {output_path}...")
    faiss.write_index_binary(index, output_path)

    del fps_array
    del index
    gc.collect()

    print(f"    Shard saved: {os.path.getsize(output_path) / 1e9:.2f} GB")


def _save_manifest(manifest_path: str, shard_files: List[str], total_compounds: int,
                   line_offsets: Optional[List[int]] = None):
    """Save manifest file for crash recovery and querying."""
    manifest = {
        'shards': shard_files,
        'total_compounds': total_compounds,
        'fingerprint_size': FINGERPRINT_SIZE,
        'nprobe': NPROBE,
        'line_offsets': line_offsets or [],
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_sharded_index(manifest_path: str, smiles_db_path: str):
    """
    Load sharded FAISS index and SMILES database.

    Returns a list of (index, offset) tuples and the database connection.
    The offset is the starting global index for each shard.
    """
    import faiss  # Lazy import: keep FAISS out of worker processes
    print(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    shard_files = manifest['shards']
    nprobe = manifest.get('nprobe', NPROBE)

    print(f"Found {len(shard_files)} shards, nprobe={nprobe}")

    # Load each shard
    indexes = []
    offset = 0

    for i, shard_path in enumerate(shard_files):
        print(f"  Loading shard {i}: {shard_path}...")
        index = faiss.read_index_binary(shard_path)
        index.nprobe = nprobe
        indexes.append((index, offset))
        offset += index.ntotal
        print(f"    {index.ntotal:,} compounds (global offset: {offset - index.ntotal:,})")

    print(f"Total compounds across all shards: {offset:,}")

    print(f"Loading SMILES database from: {smiles_db_path}")
    conn = sqlite3.connect(smiles_db_path)

    return indexes, conn


def get_smiles_by_indices(conn: sqlite3.Connection, indices: list) -> dict:
    """Retrieve SMILES strings by their FAISS indices."""
    if not indices:
        return {}
    placeholders = ','.join('?' * len(indices))
    cursor = conn.execute(
        f'SELECT idx, smiles FROM smiles WHERE idx IN ({placeholders})',
        indices
    )
    return {row[0]: row[1] for row in cursor}


def search_sharded(
    indexes: List[Tuple],
    smiles_db_conn: sqlite3.Connection,
    query_smiles: str,
    k: int = 10
) -> list:
    """
    Search for similar compounds across all shards.

    Args:
        indexes: List of (index, offset) tuples from load_sharded_index
        smiles_db_conn: SQLite database connection
        query_smiles: Query SMILES string
        k: Number of results to return

    Returns:
        List of (smiles, hamming_distance) tuples, sorted by distance
    """
    query_fp = smiles_to_binary_fingerprint(query_smiles)
    if query_fp is None:
        return []

    query_fp = query_fp.reshape(1, -1)

    # Search each shard and collect results
    all_results = []

    for index, offset in indexes:
        distances, indices = index.search(query_fp, k)

        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:
                # Convert shard-local index to global index
                global_idx = offset + int(idx)
                all_results.append((global_idx, int(dist)))

    # Sort by distance and take top k
    all_results.sort(key=lambda x: x[1])
    top_k = all_results[:k]

    # Get SMILES for top results
    global_indices = [r[0] for r in top_k]
    smiles_map = get_smiles_by_indices(smiles_db_conn, global_indices)

    results = []
    for global_idx, dist in top_k:
        if global_idx in smiles_map:
            results.append((smiles_map[global_idx], dist))

    return results


def search_sharded_lazy(
    manifest_path: str,
    smiles_db_path: str,
    query_smiles: str,
    k: int = 10
) -> list:
    """
    Search all shards without holding more than one shard in RAM at a time.

    Replaces load_sharded_index + search_sharded for large (many-shard) indexes
    where loading everything at once would exhaust available memory.  Each shard
    is loaded, searched, and freed before the next one is opened.

    Returns:
        List of (smiles, hamming_distance) tuples, sorted by distance.
    """
    import faiss

    query_fp = smiles_to_binary_fingerprint(query_smiles)
    if query_fp is None:
        return []
    query_fp = query_fp.reshape(1, -1)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    shard_files = manifest['shards']
    nprobe = manifest.get('nprobe', NPROBE)
    print(f"Lazy-searching {len(shard_files)} shards (one at a time), nprobe={nprobe}")

    all_results = []
    offset = 0

    for i, shard_path in enumerate(shard_files):
        print(f"  Shard {i}/{len(shard_files) - 1}: {shard_path}")
        index = faiss.read_index_binary(shard_path)
        index.nprobe = nprobe
        shard_total = index.ntotal

        distances, indices = index.search(query_fp, k)
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:
                all_results.append((offset + int(idx), int(dist)))

        del index
        gc.collect()
        offset += shard_total

    all_results.sort(key=lambda x: x[1])
    top_k = all_results[:k]

    conn = sqlite3.connect(smiles_db_path)
    smiles_map = get_smiles_by_indices(conn, [r[0] for r in top_k])
    conn.close()

    return [
        (smiles_map[gidx], dist)
        for gidx, dist in top_k
        if gidx in smiles_map
    ]


def search_sharded_lazy_batch(
    manifest_path: str,
    smiles_db_path: str,
    query_smiles_list: List[str],
    k: int = 10
) -> List[list]:
    """
    Search all shards for a batch of queries, loading one shard at a time.

    Avoids holding more than one shard in RAM at once (critical for indexes with
    many shards) while still amortising shard load cost across all queries.

    Returns:
        One result list per input query. Each list contains
        (smiles, hamming_distance) tuples sorted by distance. Invalid SMILES
        and queries with no hits return an empty list at the same position.
    """
    import faiss

    # Fingerprint all queries upfront; track which inputs were valid.
    fps: List[np.ndarray] = []
    valid_pos: List[int] = []
    for i, smiles in enumerate(query_smiles_list):
        fp = smiles_to_binary_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)
            valid_pos.append(i)

    if not fps:
        return [[] for _ in query_smiles_list]

    query_matrix = np.vstack(fps)  # (n_valid, FINGERPRINT_BYTES)
    n_valid = len(fps)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    shard_files = manifest['shards']
    nprobe = manifest.get('nprobe', NPROBE)
    print(f"Batch-searching {n_valid} queries across {len(shard_files)} shards "
          f"(one at a time), nprobe={nprobe}")

    # Accumulate per-query candidates across all shards.
    shard_results: List[List[Tuple[int, int]]] = [[] for _ in range(n_valid)]
    offset = 0

    for i, shard_path in enumerate(shard_files):
        print(f"  Shard {i}/{len(shard_files) - 1}...")
        index = faiss.read_index_binary(shard_path)
        index.nprobe = nprobe
        shard_total = index.ntotal

        distances, indices = index.search(query_matrix, k)
        for qi in range(n_valid):
            for idx, dist in zip(indices[qi], distances[qi]):
                if idx >= 0:
                    shard_results[qi].append((offset + int(idx), int(dist)))

        del index
        gc.collect()
        offset += shard_total

    # Resolve top-k global indices to SMILES in one DB pass per query.
    conn = sqlite3.connect(smiles_db_path)
    valid_results: List[list] = []
    for qi in range(n_valid):
        top = sorted(shard_results[qi], key=lambda x: x[1])[:k]
        smiles_map = get_smiles_by_indices(conn, [r[0] for r in top])
        valid_results.append(
            [(smiles_map[gidx], dist) for gidx, dist in top if gidx in smiles_map]
        )
    conn.close()

    # Map back to original list positions; invalid inputs keep empty lists.
    output: List[list] = [[] for _ in query_smiles_list]
    for qi, orig_i in enumerate(valid_pos):
        output[orig_i] = valid_results[qi]
    return output


def search_single_index(
    index,
    smiles_db_conn: sqlite3.Connection,
    query_smiles: str,
    k: int = 10
) -> list:
    """
    Search a single (non-sharded) FAISS index.

    For backwards compatibility with smaller indexes.
    """
    query_fp = smiles_to_binary_fingerprint(query_smiles)
    if query_fp is None:
        return []

    query_fp = query_fp.reshape(1, -1)
    distances, indices = index.search(query_fp, k)

    valid_indices = [int(i) for i in indices[0] if i >= 0]
    smiles_map = get_smiles_by_indices(smiles_db_conn, valid_indices)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= 0 and idx in smiles_map:
            results.append((smiles_map[idx], int(dist)))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Build and query sharded FAISS index for molecular similarity search (849M+ compounds)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Build sharded index from compressed SMILES file (849M compounds):
    python faiss_smiles_fingerprint.py build \\
        --input 2025.02_Enamine_REAL_HAC_24_849M_CXSMILES.cxsmiles.bz2 \\
        --output-prefix enamine_849m \\
        --smiles-db enamine_849m_smiles.db

    This creates:
      - enamine_849m_shard_00.faiss, enamine_849m_shard_01.faiss, ... (17 shards)
      - enamine_849m_smiles.db (SQLite database)
      - enamine_849m_manifest.json (for loading)

  Resume interrupted build from shard 5:
    python faiss_smiles_fingerprint.py build \\
        --input data.cxsmiles.bz2 \\
        --output-prefix enamine_849m \\
        --smiles-db enamine_849m_smiles.db \\
        --start-shard 5

  Query the sharded index:
    python faiss_smiles_fingerprint.py query \\
        --manifest enamine_849m_manifest.json \\
        --smiles-db enamine_849m_smiles.db \\
        --smiles "CCO" \\
        --k 10

  Build with compound limit (for testing):
    python faiss_smiles_fingerprint.py build \\
        --input data.cxsmiles.bz2 \\
        --output-prefix test \\
        --max-compounds 1000000

Memory usage:
  - Build: ~13-15 GB peak per shard (50M compounds × 256 bytes + overhead)
  - Query: ~13 GB per shard loaded (load shards as needed, or all for parallel search)
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build sharded FAISS index from SMILES file')
    build_parser.add_argument('--input', '-i', required=True,
                              help='Input bz2-compressed CXSMILES file')
    build_parser.add_argument('--output-prefix', '-o', default='enamine',
                              help='Prefix for output files (default: enamine)')
    build_parser.add_argument('--smiles-db', '-s', default='/lshindex/enamine_smiles.db',
                              help='Output path for SMILES SQLite database')
    build_parser.add_argument('--max-compounds', '-m', type=int, default=None,
                              help='Maximum number of compounds to process (for testing)')
    build_parser.add_argument('--start-shard', type=int, default=0,
                              help='Resume from this shard number (for crash recovery)')
    build_parser.add_argument('--num-workers', '-w', type=int, default=NUM_WORKERS,
                              help=f'Number of worker processes for fingerprint generation '
                                   f'(default: {NUM_WORKERS}, detected from CPU count)')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the sharded FAISS index')
    query_parser.add_argument('--manifest', '-m', required=True,
                              help='Path to manifest JSON file')
    query_parser.add_argument('--smiles-db', '-s', required=True,
                              help='Path to SMILES SQLite database')
    query_parser.add_argument('--smiles', '-q', required=True,
                              help='Query SMILES string')
    query_parser.add_argument('--k', '-k', type=int, default=10,
                              help='Number of results to return')

    args = parser.parse_args()

    if args.command == 'build':
        build_index(
            input_file=args.input,
            output_prefix=args.output_prefix,
            smiles_db_output=args.smiles_db,
            max_compounds=args.max_compounds,
            start_shard=args.start_shard,
            num_workers=args.num_workers
        )

    elif args.command == 'query':
        results = search_sharded_lazy(args.manifest, args.smiles_db, args.smiles, args.k)

        print(f"\nResults for query: {args.smiles}")
        print("-" * 60)
        for i, (smiles, dist) in enumerate(results, 1):
            print(f"{i:3d}. Hamming distance: {dist:4d} | {smiles}")

    else:
        parser.print_help()


if __name__ == "__main__":
    # spawn is safe on macOS/Windows: workers start a fresh interpreter and
    # only import rdkit/numpy lazily (inside smiles_to_binary_fingerprint)
    # so there is nothing heavy to initialise at startup, and no thread-lock
    # inheritance from the parent process.  On Linux fork is fine and faster.
    import sys as _sys
    if multiprocessing.get_start_method(allow_none=True) is None:
        method = 'fork' if _sys.platform == 'linux' else 'spawn'
        multiprocessing.set_start_method(method)
    main()
