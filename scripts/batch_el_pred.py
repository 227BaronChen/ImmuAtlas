#!/usr/bin/env python
"""
CLI entry point for batch EL prediction (FASTA × HLA-I alleles).

Usage examples
--------------
1) Predict a FASTA file from scratch::

     python batch_el_pred.py --fasta-file peptides.fasta

2) Import an existing EL result into cache only::

     python batch_el_pred.py \\
       --import-wide-tsv results.el_scores.wide.tsv \\
       --cache-mode write --build-cache-only

3) Predict only peptides missing from cache::

     python batch_el_pred.py \\
       --fasta-file new_input.fasta --cache-mode read --incremental

4) Show cache database statistics::

     python batch_el_pred.py \\
       --cache-stats-only --cache-db el_score_cache.sqlite --cache-mode read
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

import torch

# Allow running from repo root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from immuatlas.batch_predict import (
    DEFAULT_HLA_I,
    Checkpoint,
    ensure_output_paths,
    ensure_start_state,
    is_valid_peptide,
    iter_valid_fasta_chunks,
    load_checkpoint,
    load_model,
    load_hla_lib,
    predict_scores_for_chunk,
    print_cache_stats,
    save_checkpoint,
    validate_hlas,
    write_chunk_outputs,
)
from immuatlas.embedding import aa_dict_one_hot
from immuatlas.cache import ELScoreCache, canonical_hla_set_hash, default_model_id


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) and action.default in (False, True):
            return help_text
        if action.default is not argparse.SUPPRESS and action.default is not None:
            help_text += " (default: %(default)s)"
        return help_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch EL prediction for FASTA x HLA-I alleles",
        formatter_class=_Formatter,
        epilog=(
            "Cache notes:\n"
            "  - cache-mode=off   : no DB read/write\n"
            "  - cache-mode=read  : use cache, do not write new predictions\n"
            "  - cache-mode=write : write predictions, do not read cache\n"
            "  - cache-mode=readwrite : read cache and write new results\n"
        ),
    )
    parser.add_argument("--fasta-file", default=None, help="Input FASTA file path")
    parser.add_argument(
        "--model-path",
        default=os.path.join("data", "models", "n10_100pct.best_N6_BCELoss.bs64_lr1e-4.epoch16.pth"),
        help="EL model checkpoint path",
    )
    parser.add_argument("--hla-alleles", nargs="+", default=DEFAULT_HLA_I, help="HLA-I alleles to score")
    parser.add_argument("--batch-size", type=int, default=128, help="Prediction batch size")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Valid peptides per chunk")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--vectorized-predict", action="store_true", help="Enable vectorized prediction path")
    parser.add_argument("--output-prefix", default=None, help="Output prefix; default uses FASTA file name")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--min-len", type=int, default=None, help="Minimum peptide length filter")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum peptide length filter")

    # Cache options
    parser.add_argument("--cache-db", default="./el_score_cache.sqlite", help="SQLite cache database path")
    parser.add_argument("--cache-quant", choices=["uint16", "float32"], default="uint16", help="Score quantization mode")
    parser.add_argument("--cache-shards", default="auto", help="Shard mode: auto | single | N")
    parser.add_argument("--cache-max-shards", type=int, default=256, help="Upper bound of logical shards")
    parser.add_argument("--cache-auto-target-shard-mb", type=int, default=1024, help="Target size per shard in MB")
    parser.add_argument("--cache-mode", choices=["off", "read", "write", "readwrite"], default="off", help="Cache mode")
    parser.add_argument("--incremental", action="store_true", help="Predict only peptides missing in cache")
    parser.add_argument("--model-id", default=None, help="Cache namespace for model version")
    parser.add_argument("--import-wide-tsv", default=None, help="Import a wide EL TSV into cache")
    parser.add_argument("--import-long-tsv", default=None, help="Import a long EL TSV into cache")
    parser.add_argument("--cache-stats-only", action="store_true", help="Print cache statistics and exit")
    parser.add_argument("--stats-fasta-file", default=None, help="Optional FASTA file for cache hit-rate statistics")
    parser.add_argument("--stats-chunk-size", type=int, default=20000, help="Chunk size for stats FASTA scanning")
    parser.add_argument("--build-cache-only", action="store_true", help="Only import/build cache then exit")
    parser.add_argument("--write-long-output", action="store_true", help="Enable long-format output file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_start = time.time()
    model_id = args.model_id or default_model_id(args.model_path)
    hla_set_hash = canonical_hla_set_hash(args.hla_alleles)

    if args.incremental and args.cache_mode == "off":
        raise ValueError("--incremental requires cache mode read/readwrite")

    # ---- Cache setup ----
    cache: Optional[ELScoreCache] = None
    if args.cache_mode != "off" or args.import_wide_tsv or args.import_long_tsv or args.cache_stats_only:
        cache = ELScoreCache(
            args.cache_db,
            quant_mode=args.cache_quant,
            shards=args.cache_shards,
            max_shards=args.cache_max_shards,
            auto_target_shard_mb=args.cache_auto_target_shard_mb,
        )
        print(f"Cache DB: {args.cache_db}")
        print(f"Cache model_id: {model_id}")
        print(f"Cache quant: {args.cache_quant}")
        print(f"Cache shards: {args.cache_shards}")

        if args.cache_stats_only:
            print_cache_stats(
                cache=cache,
                model_id=model_id,
                hla_alleles=args.hla_alleles,
                hla_set_hash=hla_set_hash,
                fasta_file=args.stats_fasta_file,
                min_len=args.min_len,
                max_len=args.max_len,
                stats_chunk_size=args.stats_chunk_size,
            )
            cache.close()
            return

        if args.import_wide_tsv:
            p_count, r_count = cache.import_wide_tsv(args.import_wide_tsv, model_id=model_id, hla_alleles=args.hla_alleles)
            print(f"Imported wide TSV into cache: peptides={p_count}, rows={r_count}")

        if args.import_long_tsv:
            p_count, r_count = cache.import_long_tsv(args.import_long_tsv, model_id=model_id, hla_alleles=args.hla_alleles)
            print(f"Imported long TSV into cache: peptides={p_count}, rows={r_count}")

        if args.build_cache_only:
            cache.close()
            print("Build-cache-only finished.")
            return

    # ---- Model & library ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model_path, device)
    hla_lib = load_hla_lib()
    validate_hlas(args.hla_alleles, hla_lib)
    print(f"Predict path: {'vectorized' if args.vectorized_predict else 'classic'}")

    # ---- Output paths ----
    if args.fasta_file is None:
        print("Error: --fasta-file is required for prediction mode", file=sys.stderr)
        sys.exit(1)

    prefix = args.output_prefix or os.path.splitext(args.fasta_file)[0]
    wide_path, long_path, ckpt_path = ensure_output_paths(prefix)
    ensure_start_state(
        (wide_path, long_path, ckpt_path),
        overwrite=args.overwrite,
        resume=args.resume,
        write_long_output=args.write_long_output,
    )

    if args.resume:
        ckpt = load_checkpoint(ckpt_path)
        mode = "a"
        print(
            f"Resume enabled: start_record_index={ckpt.next_record_index}, "
            f"valid_written={ckpt.valid_written}, chunk_index={ckpt.chunk_index}"
        )
    else:
        ckpt = Checkpoint(next_record_index=0, valid_written=0, chunk_index=0, total_elapsed_sec=0.0)
        mode = "w"

    allowed_aas = set(aa_dict_one_hot.keys())

    # ---- Main prediction loop ----
    with open(wide_path, mode, encoding="utf-8", newline="") as wide_f:
        long_f = open(long_path, mode, encoding="utf-8", newline="") if args.write_long_output else None
        try:
            wide_writer = csv.writer(wide_f, delimiter="\t")
            long_writer = csv.writer(long_f, delimiter="\t") if long_f is not None else None

            if mode == "w":
                wide_writer.writerow(["peptide_id", "peptide", *args.hla_alleles])
                if long_writer is not None:
                    long_writer.writerow(["peptide_id", "peptide", "hla", "score_el"])

            total_seen = ckpt.next_record_index
            for ids, peptides, seen_after_chunk, skipped in iter_valid_fasta_chunks(
                args.fasta_file,
                args.chunk_size,
                allowed_aas,
                start_record_index=ckpt.next_record_index,
                min_len=args.min_len,
                max_len=args.max_len,
            ):
                chunk_start = time.time()
                hit_count = 0
                miss_unique_count = 0

                cached_by_peptide: Dict[str, Dict[str, float]] = {}
                if args.incremental and cache is not None and args.cache_mode in ["read", "readwrite"]:
                    cached_by_peptide = cache.lookup_scores(
                        peptides=peptides,
                        hla_alleles=args.hla_alleles,
                        model_id=model_id,
                        hla_set_hash=hla_set_hash,
                    )

                if args.incremental:
                    miss_unique_peptides: List[str] = []
                    seen_peptides = set()
                    for pep in peptides:
                        cached_scores = cached_by_peptide.get(pep, {})
                        if all(h in cached_scores for h in args.hla_alleles):
                            hit_count += 1
                            continue
                        if pep not in seen_peptides:
                            seen_peptides.add(pep)
                            miss_unique_peptides.append(pep)

                    miss_unique_count = len(miss_unique_peptides)

                    pred_scores_by_hla: Dict[str, List[float]] = {h: [] for h in args.hla_alleles}
                    pred_by_peptide: Dict[str, Dict[str, float]] = {}
                    if miss_unique_peptides:
                        pred_scores_by_hla = predict_scores_for_chunk(
                            model=model,
                            peptides=miss_unique_peptides,
                            hla_alleles=args.hla_alleles,
                            hla_lib=hla_lib,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            vectorized_predict=args.vectorized_predict,
                        )
                        for i, pep in enumerate(miss_unique_peptides):
                            pred_by_peptide[pep] = {h: float(pred_scores_by_hla[h][i]) for h in args.hla_alleles}

                    if cache is not None and args.cache_mode in ["write", "readwrite"] and miss_unique_peptides:
                        cache.upsert_scores(
                            peptides=miss_unique_peptides,
                            hla_alleles=args.hla_alleles,
                            scores_by_hla=pred_scores_by_hla,
                            model_id=model_id,
                            hla_set_hash=hla_set_hash,
                        )

                    scores_by_hla = {h: [] for h in args.hla_alleles}
                    for pep in peptides:
                        cached_scores = cached_by_peptide.get(pep, {})
                        if all(h in cached_scores for h in args.hla_alleles):
                            for h in args.hla_alleles:
                                scores_by_hla[h].append(float(cached_scores[h]))
                        else:
                            for h in args.hla_alleles:
                                scores_by_hla[h].append(float(pred_by_peptide[pep][h]))
                else:
                    scores_by_hla = predict_scores_for_chunk(
                        model=model,
                        peptides=peptides,
                        hla_alleles=args.hla_alleles,
                        hla_lib=hla_lib,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        vectorized_predict=args.vectorized_predict,
                    )
                    if cache is not None and args.cache_mode in ["write", "readwrite"]:
                        cache.upsert_scores(
                            peptides=peptides,
                            hla_alleles=args.hla_alleles,
                            scores_by_hla=scores_by_hla,
                            model_id=model_id,
                            hla_set_hash=hla_set_hash,
                        )

                write_chunk_outputs(
                    ids=ids,
                    peptides=peptides,
                    hla_alleles=args.hla_alleles,
                    scores_by_hla=scores_by_hla,
                    wide_writer=wide_writer,
                    long_writer=long_writer,
                )

                wide_f.flush()
                if long_f is not None:
                    long_f.flush()

                ckpt.next_record_index = seen_after_chunk
                ckpt.valid_written += len(peptides)
                ckpt.chunk_index += 1
                ckpt.total_elapsed_sec += time.time() - chunk_start
                total_seen = seen_after_chunk

                save_checkpoint(
                    ckpt_path=ckpt_path,
                    ckpt=ckpt,
                    fasta_file=args.fasta_file,
                    model_path=args.model_path,
                    hla_alleles=args.hla_alleles,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                )

                elapsed = time.time() - chunk_start
                throughput = len(peptides) / elapsed if elapsed > 0 else 0.0
                if args.incremental:
                    print(
                        f"Chunk {ckpt.chunk_index}: valid={len(peptides)}, hit={hit_count}, miss_unique={miss_unique_count}, "
                        f"total_seen={total_seen}, skipped_on_resume={skipped}, elapsed={elapsed:.1f}s, throughput={throughput:.1f} pep/s"
                    )
                else:
                    print(
                        f"Chunk {ckpt.chunk_index}: valid={len(peptides)}, total_seen={total_seen}, "
                        f"skipped_on_resume={skipped}, elapsed={elapsed:.1f}s, throughput={throughput:.1f} pep/s"
                    )
        finally:
            if long_f is not None:
                long_f.close()

    total_elapsed = time.time() - run_start
    print(
        f"Done. valid_written={ckpt.valid_written}, total_seen={ckpt.next_record_index}, "
        f"runtime={total_elapsed:.1f}s"
    )
    print(f"Wide output: {wide_path}")
    if args.write_long_output:
        print(f"Long output: {long_path}")
    else:
        print("Long output: disabled")
    print(f"Checkpoint: {ckpt_path}")

    if cache is not None:
        cache.close()


if __name__ == "__main__":
    main()
