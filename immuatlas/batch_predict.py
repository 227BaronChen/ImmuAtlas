"""
Batch EL prediction for FASTA × HLA-I alleles.

Provides the core prediction logic with optional SQLite caching, checkpointing,
and vectorized GPU inference.  The CLI entry point is in
``scripts/batch_el_pred.py``.
"""

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import pandas as pd
from Bio import SeqIO
from torch.utils.data import DataLoader

from immuatlas.model import ELBaseModel
from immuatlas.data_loader import PeptideHLADataset2
from immuatlas.embedding import aa_dict_one_hot, load_hla_library
from immuatlas.cache import ELScoreCache, canonical_hla_set_hash, default_model_id

# ---------------------------------------------------------------------------
# Default HLA-I allele list (China top 10)
# ---------------------------------------------------------------------------

DEFAULT_HLA_I = [
    "A1101", "A2402", "C0702", "C0102", "A0201",
    "B4001", "C0304", "B4601", "A0207", "A3303",
]


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    next_record_index: int
    valid_written: int
    chunk_index: int
    total_elapsed_sec: float


# ---------------------------------------------------------------------------
# Model / library loaders
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> ELBaseModel:
    """Load the ELBaseModel from a checkpoint file."""
    model = ELBaseModel(d_model=320, nhead=8, num_encoder_layers=6, dim_feedforward=512)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_hla_lib(hla_library_file: str = None) -> Dict[str, str]:
    """Load HLA library from file (defaults to bundled library)."""
    from immuatlas.embedding import load_hla_library as _load
    return _load(hla_library_file)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_hlas(hla_alleles: Sequence[str], hla_lib: Dict[str, str]) -> None:
    missing = [hla for hla in hla_alleles if hla not in hla_lib]
    if missing:
        raise ValueError(f"HLA alleles not found in library: {missing}")


def is_valid_peptide(seq: str, allowed_aas: set, min_len: int = None, max_len: int = None) -> bool:
    if not seq:
        return False
    if min_len is not None and len(seq) < min_len:
        return False
    if max_len is not None and len(seq) > max_len:
        return False
    return all(aa in allowed_aas for aa in seq)


# ---------------------------------------------------------------------------
# FASTA iteration
# ---------------------------------------------------------------------------

def iter_valid_fasta_chunks(
    fasta_path: str,
    chunk_size: int,
    allowed_aas: set,
    start_record_index: int,
    min_len: int = None,
    max_len: int = None,
) -> Iterator[Tuple[List[str], List[str], int, int]]:
    """Yield (ids, peptides, total_seen, skipped) tuples from a FASTA file."""
    chunk_ids: List[str] = []
    chunk_peptides: List[str] = []

    total_seen = 0
    skipped = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        if total_seen < start_record_index:
            total_seen += 1
            skipped += 1
            continue

        total_seen += 1
        seq = str(record.seq).upper()
        if is_valid_peptide(seq, allowed_aas, min_len=min_len, max_len=max_len):
            chunk_ids.append(record.id)
            chunk_peptides.append(seq)

            if len(chunk_peptides) >= chunk_size:
                yield chunk_ids, chunk_peptides, total_seen, skipped
                chunk_ids = []
                chunk_peptides = []

    if chunk_peptides:
        yield chunk_ids, chunk_peptides, total_seen, skipped


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_scores_for_chunk(
    model: ELBaseModel,
    peptides: Sequence[str],
    hla_alleles: Sequence[str],
    hla_lib: Dict[str, str],
    batch_size: int,
    num_workers: int = 0,
    vectorized_predict: bool = False,
) -> Dict[str, List[float]]:
    """Predict EL scores for a chunk of peptides × HLA alleles."""
    if not peptides:
        return {h: [] for h in hla_alleles}

    if vectorized_predict:
        device = next(model.parameters()).device
        peptide_tensors = torch.stack([model._encode_peptide(p) for p in peptides]).to(device)
        hla_tensors = torch.stack([model._encode_hla(hla_lib[h]) for h in hla_alleles]).to(device)

        pep_count = len(peptides)
        hla_count = len(hla_alleles)
        total_pairs = pep_count * hla_count
        scores_flat = torch.empty(total_pairs, dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, total_pairs, batch_size):
                end = min(start + batch_size, total_pairs)
                pair_idx = torch.arange(start, end, device=device)
                pep_idx = torch.div(pair_idx, hla_count, rounding_mode="floor")
                hla_idx = pair_idx % hla_count

                pep_batch = peptide_tensors.index_select(0, pep_idx)
                hla_batch = hla_tensors.index_select(0, hla_idx)
                preds = model(peptide_tensors=pep_batch, hla_tensors=hla_batch)
                scores_flat[start:end] = preds.squeeze(-1).detach().cpu()

        score_matrix = scores_flat.view(pep_count, hla_count)
        return {hla: score_matrix[:, i].tolist() for i, hla in enumerate(hla_alleles)}

    # Classic path
    scores_by_hla: Dict[str, List[float]] = {}

    with torch.no_grad():
        for hla in hla_alleles:
            hla_seq = hla_lib[hla]
            hla_seqs = [hla_seq] * len(peptides)
            dataset = PeptideHLADataset2(list(peptides), hla_seqs)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            hla_scores: List[float] = []
            for batch_peptides, batch_hla_seqs in dataloader:
                preds = model(batch_peptides, hla_seq_list=batch_hla_seqs)
                hla_scores.extend(preds.squeeze(-1).detach().cpu().tolist())

            scores_by_hla[hla] = hla_scores

    return scores_by_hla


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def ensure_output_paths(prefix: str) -> Tuple[str, str, str]:
    wide_path = f"{prefix}.el_scores.wide.tsv"
    long_path = f"{prefix}.el_scores.long.tsv"
    ckpt_path = f"{prefix}.el_scores.ckpt.json"
    return wide_path, long_path, ckpt_path


def write_chunk_outputs(
    ids: Sequence[str],
    peptides: Sequence[str],
    hla_alleles: Sequence[str],
    scores_by_hla: Dict[str, List[float]],
    wide_writer: csv.writer,
    long_writer: Optional[csv.writer] = None,
) -> None:
    for i, (pid, pep) in enumerate(zip(ids, peptides)):
        wide_row = [pid, pep]
        for hla in hla_alleles:
            score = scores_by_hla[hla][i]
            wide_row.append(f"{score:.8f}")
            if long_writer is not None:
                long_writer.writerow([pid, pep, hla, f"{score:.8f}"])
        wide_writer.writerow(wide_row)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_path: str,
    ckpt: Checkpoint,
    fasta_file: str,
    model_path: str,
    hla_alleles: Sequence[str],
    batch_size: int,
    chunk_size: int,
) -> None:
    payload = {
        "fasta_file": fasta_file,
        "model_path": model_path,
        "hla_alleles": list(hla_alleles),
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "next_record_index": ckpt.next_record_index,
        "valid_written": ckpt.valid_written,
        "chunk_index": ckpt.chunk_index,
        "total_elapsed_sec": ckpt.total_elapsed_sec,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_checkpoint(ckpt_path: str) -> Checkpoint:
    with open(ckpt_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return Checkpoint(
        next_record_index=int(payload.get("next_record_index", 0)),
        valid_written=int(payload.get("valid_written", 0)),
        chunk_index=int(payload.get("chunk_index", 0)),
        total_elapsed_sec=float(payload.get("total_elapsed_sec", 0.0)),
    )


def ensure_start_state(
    paths: Tuple[str, str, str],
    overwrite: bool,
    resume: bool,
    write_long_output: bool,
) -> None:
    wide_path, long_path, ckpt_path = paths
    required_paths = [wide_path, ckpt_path]
    if write_long_output:
        required_paths.append(long_path)

    if resume:
        if not all(os.path.exists(p) for p in required_paths):
            raise FileNotFoundError("Resume requested, but output files/checkpoint are missing.")
        return

    existing = [p for p in required_paths if os.path.exists(p)]
    if existing and not overwrite:
        raise FileExistsError(f"Output files exist: {existing}. Use --overwrite or --resume.")

    if overwrite:
        for p in required_paths:
            if os.path.exists(p):
                os.remove(p)


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------

def print_cache_stats(
    cache: ELScoreCache,
    model_id: str,
    hla_alleles: Sequence[str],
    hla_set_hash: str,
    fasta_file: str = None,
    min_len: int = None,
    max_len: int = None,
    stats_chunk_size: int = 20000,
) -> None:
    """Print cache database statistics."""
    stats = cache.stats()
    file_stats = cache.file_sizes()
    shard_info = cache.shard_overview() if hasattr(cache, "shard_overview") else None

    print(f"Cache DB: {cache.db_path}")
    print(f"Peptides in cache: {stats['peptides']}")
    print(f"Score rows in cache: {stats['scores']}")
    print(f"Distinct models: {stats['models']}")
    print(f"Distinct HLA alleles: {stats['hlas']}")
    print()

    def format_bytes(nbytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if nbytes < 1024:
                return f"{nbytes:.1f}{unit}"
            nbytes /= 1024
        return f"{nbytes:.1f}TB"

    main_size_str = format_bytes(file_stats['main_db_bytes'])
    wal_size_str = format_bytes(file_stats['wal_bytes'])
    shm_size_str = format_bytes(file_stats['shm_bytes'])
    total_size_str = format_bytes(file_stats['total_bytes'])

    print("Database File Sizes:")
    print(f"  Main DB: {main_size_str}")
    if file_stats['wal_bytes'] > 0:
        print(f"  WAL:     {wal_size_str}")
    if file_stats['shm_bytes'] > 0:
        print(f"  SHM:     {shm_size_str}")
    print(f"  Total:   {total_size_str}")
    print()

    print("Database Pages:")
    print(f"  Total pages: {file_stats['page_count']} × {file_stats['page_size']}B")
    print(f"  Used size: {format_bytes(file_stats['used_bytes'])}")
    print(f"  Index ~{file_stats['index_percentage']:.1f}% of data")
    print()

    print(f"Model ID: {model_id}")
    print(f"HLA set hash: {hla_set_hash}")

    if shard_info is not None:
        print()
        print("Shard Overview:")
        print(f"  Mode: {shard_info.get('mode', 'unknown')}")
        print(f"  Requested: {shard_info.get('requested', '')}")
        print(f"  Logical shards: {shard_info.get('logical_shards', 0)}")
        print(f"  Active shards: {shard_info.get('active_shards', 0)}")
        if shard_info.get("shard_dir"):
            print(f"  Shard dir: {shard_info['shard_dir']}")
        print(
            f"  Size distribution: min={format_bytes(shard_info.get('bytes_min', 0))}, "
            f"max={format_bytes(shard_info.get('bytes_max', 0))}, "
            f"avg={format_bytes(shard_info.get('bytes_avg', 0))}"
        )
        top_shards = shard_info.get("top_shards", [])
        if top_shards:
            print("  Top shards by size:")
            for row in top_shards:
                print(
                    f"    shard_{int(row['shard']):04d}: total={format_bytes(row.get('total_bytes', 0))}, "
                    f"main={format_bytes(row.get('main_db_bytes', 0))}, "
                    f"wal={format_bytes(row.get('wal_bytes', 0))}"
                )
    print()

    if fasta_file:
        hit_set = cache.full_hit_set_for_namespace(
            model_id=model_id,
            hla_set_hash=hla_set_hash,
            hla_alleles=hla_alleles,
        )
        allowed_aas = set(aa_dict_one_hot.keys())
        total_records = 0
        hit_records = 0
        for _, peptides, _, _ in iter_valid_fasta_chunks(
            fasta_file,
            stats_chunk_size,
            allowed_aas,
            start_record_index=0,
            min_len=min_len,
            max_len=max_len,
        ):
            total_records += len(peptides)
            hit_records += sum(1 for pep in peptides if pep in hit_set)

        hit_rate = (hit_records / total_records) if total_records else 0.0
        print(f"Input FASTA: {fasta_file}")
        print(f"Input valid peptides: {total_records}")
        print(f"Hit peptides: {hit_records}")
        print(f"Miss peptides: {total_records - hit_records}")
        print(f"Hit rate: {hit_rate:.2%}")
