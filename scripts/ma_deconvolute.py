#!/usr/bin/env python
"""
CLI entry point for MA deconvolution EL prediction.

Read peptides from a FASTA file, predict EL scores against HLA-I alleles,
deconvolute which allele presents each peptide, and output ranked results.

Usage examples
--------------
1) Specify HLA alleles on the command line::

     python ma_deconvolute.py --fasta-file input.fasta \\
       --hla-alleles A3101,A2402,B3543,B5101,C0102,C1502

2) Read HLA alleles from a file::

     python ma_deconvolute.py --fasta-file input.fasta \\
       --hla-file hla_alleles.txt

3) No cache, custom output prefix::

     python ma_deconvolute.py --fasta-file input.fasta \\
       --hla-alleles A1101 --cache-mode off \\
       --output-prefix my_result

4) Custom top-pct filters::

     python ma_deconvolute.py --fasta-file input.fasta \\
       --hla-alleles A3101,A2402 --top-pct 1,5,10
"""

import argparse
import os
import sys
import time
from typing import List

import torch

# Allow running from repo root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from immuatlas.cache import ELScoreCache, default_model_id
from immuatlas.ma_deconvolution import (
    filter_by_rank,
    read_fasta_and_predict,
)
from immuatlas.model import ELBaseModel


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
        description="MA deconvolution EL prediction: read peptides from FASTA, "
        "predict EL scores against HLA-I alleles, and deconvolute.",
        formatter_class=_Formatter,
        epilog=(
            "Examples:\n"
            "  1) Specify HLA alleles on the command line:\n"
            "     python ma_deconvolute.py --fasta-file input.fasta \\\n"
            "       --hla-alleles A3101,A2402,B3543,B5101,C0102,C1502\n\n"
            "  2) Read HLA alleles from a file:\n"
            "     python ma_deconvolute.py --fasta-file input.fasta \\\n"
            "       --hla-file hla_alleles.txt\n\n"
            "  3) No cache, custom output prefix:\n"
            "     python ma_deconvolute.py --fasta-file input.fasta \\\n"
            "       --hla-alleles A1101 --cache-mode off \\\n"
            "       --output-prefix my_result\n\n"
            "  4) Custom top-pct filters:\n"
            "     python ma_deconvolute.py --fasta-file input.fasta \\\n"
            "       --hla-alleles A3101,A2402 --top-pct 1,5,10\n\n"
            "Cache notes:\n"
            "  - cache-mode=off       : no DB read/write\n"
            "  - cache-mode=read      : use cache, do not write new predictions\n"
            "  - cache-mode=write     : write predictions, do not read cache\n"
            "  - cache-mode=readwrite : read cache and write new results\n"
        ),
    )

    # ---- Input / output ----
    parser.add_argument("--fasta-file", required=True, help="Input FASTA file path")
    parser.add_argument(
        "--model-path",
        default=os.path.join("data", "models", "n10_100pct.best_N6_BCELoss.bs64_lr1e-4.epoch16.pth"),
        help="EL model checkpoint path",
    )
    parser.add_argument(
        "--hla-alleles",
        nargs="+",
        default=["A1101"],
        help="HLA-I alleles (space or comma separated, e.g. A3101,A2402 or A3101 A2402)",
    )
    parser.add_argument(
        "--hla-file",
        default=None,
        help="Read HLA alleles from file (one per line); can be combined with --hla-alleles",
    )

    # ---- Deconvolution parameters ----
    parser.add_argument(
        "--method",
        choices=["simple", "optimized"],
        default="optimized",
        help="Deconvolution method",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Absolute EL score threshold (T); 0 enables percentile mode",
    )
    parser.add_argument(
        "--difference",
        type=float,
        default=0.1,
        help="Significant advantage threshold (D), optimized method only",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.15,
        help="Multi-assignment threshold (G), optimized method only",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Prediction batch size")

    # ---- Output control ----
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix; derived from FASTA filename if not specified",
    )
    parser.add_argument(
        "--top-pct",
        type=str,
        nargs="+",
        default=["3", "10", "30"],
        help="Percentile filter thresholds (space or comma separated, e.g. 3,10,30)",
    )

    # ---- Cache ----
    parser.add_argument("--cache-db", default=None, help="SQLite cache database path")
    parser.add_argument(
        "--cache-quant",
        choices=["uint16", "float32"],
        default="uint16",
        help="Cache score quantization mode",
    )
    parser.add_argument(
        "--cache-shards",
        default="auto",
        help="Shard mode: auto | single | N",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["off", "read", "write", "readwrite"],
        default="read",
        help="Cache mode",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Cache namespace model identifier; auto-derived from checkpoint if not specified",
    )

    args = parser.parse_args()

    # ---- Flatten comma-separated values ----
    if args.hla_alleles:
        flat: List[str] = []
        for item in args.hla_alleles:
            flat.extend(part.strip() for part in item.split(",") if part.strip())
        args.hla_alleles = flat

    flat_pct: List[float] = []
    for item in args.top_pct:
        flat_pct.extend(float(part.strip()) for part in item.split(",") if part.strip())
    args.top_pct = flat_pct

    # ---- Validation ----
    if not args.hla_alleles and args.hla_file is None:
        parser.error("At least one of --hla-alleles or --hla-file is required")

    if args.hla_file is not None:
        file_hlas: List[str] = []
        with open(args.hla_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    file_hlas.append(line)
        seen = set(args.hla_alleles)
        for h in file_hlas:
            if h not in seen:
                args.hla_alleles.append(h)
                seen.add(h)

    if args.output_prefix is None:
        args.output_prefix = os.path.splitext(os.path.basename(args.fasta_file))[0]

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()
    args = parse_args()

    # ---- Load model ----
    model = ELBaseModel(
        d_model=320, nhead=8, num_encoder_layers=6, dim_feedforward=512
    )
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded from: {args.model_path}")
    print(f"Device: {device}")

    # ---- Cache ----
    model_id = args.model_id or default_model_id(args.model_path)
    if args.cache_db is not None:
        cache = ELScoreCache(
            args.cache_db,
            quant_mode=args.cache_quant,
            shards=args.cache_shards,
        )
        cache_mode = args.cache_mode
        print(f"Cache DB: {args.cache_db}")
        print(f"Cache model_id: {model_id}")
        print(f"Cache mode: {cache_mode}")
    else:
        cache = None
        cache_mode = "off"
        print("Cache: disabled (no --cache-db specified)")

    # ---- HLA alleles ----
    print(f"HLA alleles ({len(args.hla_alleles)}): {args.hla_alleles}")

    # ---- Predict & deconvolute ----
    results_df = read_fasta_and_predict(
        model=model,
        fasta_file=args.fasta_file,
        hla_alleles=args.hla_alleles,
        method=args.method,
        threshold=args.threshold,
        difference=args.difference,
        gap=args.gap,
        batch_size=args.batch_size,
        cache=cache,
        model_id=model_id,
        cache_mode=cache_mode,
    )

    # ---- Save full results ----
    all_tsv_path = f"{args.output_prefix}.ImmuAtlas_EL_pred.all.tsv"
    results_df.to_csv(all_tsv_path, sep="\t", index=False)
    print(f"Full results saved to: {all_tsv_path}")

    # ---- Save top-pct filtered FASTA files ----
    for pct in args.top_pct:
        results_df_top = filter_by_rank(results_df, max_rank_pct=pct)
        pct_str = str(int(pct)) if pct == int(pct) else str(pct)
        fasta_path = f"{args.output_prefix}.ImmuAtlas_EL_pred.top_{pct_str}pct.fasta"
        with open(fasta_path, "w") as f:
            for _, row in results_df_top.iterrows():
                f.write(f">{row['peptide_id']}\n{row['peptide']}\n")
        print(f"Top {pct_str}% ({len(results_df_top)} peptides) saved to: {fasta_path}")

    if cache is not None:
        cache.close()

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
