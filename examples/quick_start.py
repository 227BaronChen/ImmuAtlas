#!/usr/bin/env python
"""
Quick-start demo for ImmuAtlas.

Run from the repo root:
    python examples/quick_start.py
"""

import os
import sys

# Ensure the package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from immuatlas import PeptideMHCPredictor, MAELDeconvolution


def main():
    # ------------------------------------------------------------------
    # 1. Single-pair prediction
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. Single-pair EL prediction")
    print("=" * 60)

    predictor = PeptideMHCPredictor()

    result = predictor.predict("LLGATCMV", "HLA-A*02:01")
    print(f"  Peptide : {result.peptide}")
    print(f"  HLA     : {result.hla}")
    print(f"  Score_EL: {result.score_el:.4f}")
    print()

    # ------------------------------------------------------------------
    # 2. Batch prediction
    # ------------------------------------------------------------------
    print("=" * 60)
    print("2. Batch EL prediction")
    print("=" * 60)

    pairs = [
        ("LLGATCMV", "HLA-A*02:01"),
        ("KVLEYVIK", "HLA-B*07:02"),
        ("YLNDHLEPWI", "HLA-A*02:01"),
        ("GILGFVFTL", "HLA-A*02:01"),
    ]
    results = predictor.predict_batch(pairs)
    for r in results:
        print(f"  {r.peptide:12s} × {r.hla:15s} → Score_EL = {r.score_el:.4f}")
    print()

    # ------------------------------------------------------------------
    # 3. MA Deconvolution (simple)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("3. MA Deconvolution – simple (winner-take-all)")
    print("=" * 60)

    deconv = MAELDeconvolution(predictor.model)
    peptide_hla_pairs = [
        ("LLGATCMV", ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*11:01"]),
        ("KVLEYVIK", ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*11:01"]),
        ("GILGFVFTL", ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*11:01"]),
    ]
    simple_results = deconv.deconvolute_simple(peptide_hla_pairs, threshold=0.5)
    for r in simple_results:
        print(f"  {r['peptide']:12s} → assigned: {r['assigned_hla']} ({r['assigned_score']:.4f})")
    print()

    # ------------------------------------------------------------------
    # 4. MA Deconvolution (optimized)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("4. MA Deconvolution – optimized (multi-restriction)")
    print("=" * 60)

    opt_results = deconv.deconvolute_optimized(
        peptide_hla_pairs, threshold=0.5, difference=0.1, gap=0.15
    )
    for r in opt_results:
        hlas_str = ", ".join(r["assigned_hlas"]) if r["assigned_hlas"] else "None"
        scores_str = ", ".join(f"{s:.4f}" for s in r["assigned_scores"]) if r["assigned_scores"] else "N/A"
        print(f"  {r['peptide']:12s} → assigned: [{hlas_str}] scores: [{scores_str}]")
    print()

    print("Done!")


if __name__ == "__main__":
    main()
