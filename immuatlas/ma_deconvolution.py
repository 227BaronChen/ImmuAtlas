"""
MA (Multiple Allele) deconvolution for HLA peptide presentation prediction.

Given a set of peptides and the patient's HLA-I alleles, this module
deconvolutes which HLA allele is most likely responsible for presenting
each peptide, using EL (Expression Level) scores from the base model.

Two methods are provided:
  - **simple** (winner-take-all): assign each peptide to the HLA with
    the highest EL score, if above threshold.
  - **optimized** (multi-restriction): allow multi-HLA assignment when
    scores are close, controlled by T / D / G thresholds.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from immuatlas.cache import ELScoreCache, canonical_hla_set_hash, default_model_id
from immuatlas.embedding import aa_dict_one_hot, aa_dict_tokens, load_hla_library
from immuatlas.model import ELBaseModel


# ---------------------------------------------------------------------------
# MAELDeconvolution class
# ---------------------------------------------------------------------------


class MAELDeconvolution:
    """
    Deconvolute MA (Multiple Allele) mass-spectrometry data using the EL
    base model.  Supports two methods:

    * **simple** – winner-take-all: each peptide is assigned to the single
      HLA allele with the highest EL score (if above threshold).
    * **optimized** – multi-restriction: a peptide may be assigned to
      multiple HLA alleles when their EL scores are close, governed by
      threshold (T), difference (D), and gap (G) parameters.
    """

    def __init__(
        self,
        model: ELBaseModel,
        hla_library_file: Optional[str] = None,
        cache: Optional[ELScoreCache] = None,
        model_id: Optional[str] = None,
        cache_mode: str = "readwrite",
    ):
        """
        Parameters
        ----------
        model : ELBaseModel
            Trained EL base model.
        hla_library_file : str or None
            Path to HLA pseudo-sequence library TSV.  ``None`` uses the
            bundled ``data/HLA_library3.tsv``.
        cache : ELScoreCache or None
            EL score cache instance.  ``None`` disables caching.
        model_id : str or None
            Model identifier for cache namespace.
        cache_mode : str
            ``'off'`` | ``'read'`` | ``'write'`` | ``'readwrite'``.
        """
        self.model = model
        self.model.eval()

        # Cache
        self.cache = cache
        self.model_id = model_id
        self.cache_mode = cache_mode if cache is not None else "off"

        # Load HLA library
        self.hla_lib = load_hla_library(hla_library_file)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_encode_peptides(
        peptides: List[str], max_len: int = 25
    ) -> torch.Tensor:
        """Vectorised batch one-hot encoding of peptides."""
        n = len(peptides)
        arr = np.zeros((n, max_len, 21), dtype=np.float32)
        for i, pep in enumerate(peptides):
            for j, aa in enumerate(pep):
                arr[i, j, aa_dict_tokens.get(aa, 20)] = 1.0
            for j in range(len(pep), max_len):
                arr[i, j, 20] = 1.0
        return torch.from_numpy(arr)

    @staticmethod
    def _batch_encode_hlas(hla_seqs: List[str]) -> torch.Tensor:
        """Vectorised batch one-hot encoding of HLA pseudo-sequences."""
        max_len = max(len(s) for s in hla_seqs)
        n = len(hla_seqs)
        arr = np.zeros((n, max_len, 21), dtype=np.float32)
        for i, seq in enumerate(hla_seqs):
            for j, aa in enumerate(seq):
                arr[i, j, aa_dict_tokens.get(aa, 20)] = 1.0
        return torch.from_numpy(arr)

    # ------------------------------------------------------------------
    # Prediction (with cache)
    # ------------------------------------------------------------------

    def _predict_el_scores(
        self,
        peptides: List[str],
        hla_alleles: List[str],
        batch_size: int = 64,
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict EL scores for *peptides × HLA alleles*.

        When caching is enabled, already-cached scores are read first and
        only missing peptides are predicted on GPU, then written back.
        """
        # ---- Cache path ----
        if self.cache is not None and self.cache_mode != "off":
            hla_set_hash = canonical_hla_set_hash(hla_alleles)
            cached_by_peptide: Dict[str, Dict[str, float]] = {}
            hit_count = 0

            if self.cache_mode in ("read", "readwrite"):
                cached_by_peptide = self.cache.lookup_scores(
                    peptides=peptides,
                    hla_alleles=hla_alleles,
                    model_id=self.model_id,
                    hla_set_hash=hla_set_hash,
                )
                for pep in peptides:
                    cached_scores = cached_by_peptide.get(pep, {})
                    if all(h in cached_scores for h in hla_alleles):
                        hit_count += 1

            miss_peptides: List[str] = []
            seen = set()
            for pep in peptides:
                cached_scores = cached_by_peptide.get(pep, {})
                if not all(h in cached_scores for h in hla_alleles):
                    if pep not in seen:
                        seen.add(pep)
                        miss_peptides.append(pep)

            miss_count = len(miss_peptides)
            print(
                f"Cache: {hit_count}/{len(peptides)} peptides hit, "
                f"{miss_count} unique peptides to predict"
            )

            miss_results: Dict[str, Dict[str, float]] = {}
            if miss_peptides:
                miss_results = self._predict_el_scores_gpu(
                    miss_peptides, hla_alleles, batch_size
                )
                if self.cache_mode in ("write", "readwrite"):
                    scores_by_hla: Dict[str, List[float]] = {
                        h: [] for h in hla_alleles
                    }
                    for pep in miss_peptides:
                        for h in hla_alleles:
                            scores_by_hla[h].append(miss_results[pep][h])
                    self.cache.upsert_scores(
                        peptides=miss_peptides,
                        hla_alleles=hla_alleles,
                        scores_by_hla=scores_by_hla,
                        model_id=self.model_id,
                        hla_set_hash=hla_set_hash,
                    )

            results: Dict[str, Dict[str, float]] = {}
            for pep in peptides:
                cached_scores = cached_by_peptide.get(pep, {})
                if all(h in cached_scores for h in hla_alleles):
                    results[pep] = {h: float(cached_scores[h]) for h in hla_alleles}
                else:
                    results[pep] = miss_results[pep]
            return results

        # ---- No-cache path ----
        return self._predict_el_scores_gpu(peptides, hla_alleles, batch_size)

    def _predict_el_scores_gpu(
        self,
        peptides: List[str],
        hla_alleles: List[str],
        batch_size: int = 64,
    ) -> Dict[str, Dict[str, float]]:
        """Pure GPU vectorised prediction (no cache logic)."""
        device = next(self.model.parameters()).device
        pep_count = len(peptides)
        hla_count = len(hla_alleles)
        total_pairs = pep_count * hla_count

        # Pre-encode all peptides and HLA sequences
        hla_seqs = [self.hla_lib[h] for h in hla_alleles]
        peptide_tensors = self._batch_encode_peptides(peptides).to(device)
        hla_tensors = self._batch_encode_hlas(hla_seqs).to(device)

        # Pre-compute pair indices on GPU
        all_pep_idx = torch.arange(pep_count, device=device).repeat_interleave(
            hla_count
        )
        all_hla_idx = torch.arange(hla_count, device=device).repeat(pep_count)

        # Vectorised prediction – collect on GPU, copy to CPU once
        scores_gpu = torch.empty(total_pairs, dtype=torch.float32, device=device)

        with torch.no_grad():
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
            ):
                for start in range(0, total_pairs, batch_size):
                    end = min(start + batch_size, total_pairs)
                    pep_batch = peptide_tensors[all_pep_idx[start:end]]
                    hla_batch = hla_tensors[all_hla_idx[start:end]]
                    preds = self.model(
                        peptide_tensors=pep_batch, hla_tensors=hla_batch
                    )
                    scores_gpu[start:end] = preds.squeeze(-1)

        score_matrix = scores_gpu.view(pep_count, hla_count).cpu().numpy()
        results = {}
        for i, peptide in enumerate(peptides):
            results[peptide] = {
                hla_alleles[j]: float(score_matrix[i, j])
                for j in range(hla_count)
            }
        return results

    # ------------------------------------------------------------------
    # Deconvolution methods
    # ------------------------------------------------------------------

    def deconvolute_simple(
        self,
        peptide_hla_pairs: List[Tuple[str, List[str]]],
        threshold: float = 0.5,
        batch_size: int = 64,
    ) -> List[Dict]:
        """
        Simple (winner-take-all) deconvolution.

        Each peptide is assigned to the HLA allele with the highest EL
        score, provided that score exceeds *threshold*.

        Parameters
        ----------
        peptide_hla_pairs : list of (peptide, [hla1, hla2, ...])
        threshold : float
            Absolute EL score threshold.
        batch_size : int

        Returns
        -------
        list of dict with keys ``peptide``, ``assigned_hla``,
        ``assigned_score``, ``all_scores``.
        """
        results: List[Dict] = []

        all_peptides = list({pair[0] for pair in peptide_hla_pairs})
        all_hla_alleles = list(
            {h for _, hlas in peptide_hla_pairs for h in hlas}
        )

        all_el_scores = self._predict_el_scores(
            all_peptides, all_hla_alleles, batch_size
        )

        for peptide, hla_alleles in peptide_hla_pairs:
            el_scores = {
                hla: all_el_scores[peptide][hla] for hla in hla_alleles
            }
            best_hla = max(el_scores, key=el_scores.get)
            best_score = el_scores[best_hla]

            if best_score < threshold:
                result = {
                    "peptide": peptide,
                    "assigned_hla": "None",
                    "assigned_score": 0.0,
                    "all_scores": el_scores,
                }
            else:
                result = {
                    "peptide": peptide,
                    "assigned_hla": best_hla,
                    "assigned_score": best_score,
                    "all_scores": el_scores,
                }
            results.append(result)

        return results

    def deconvolute_optimized(
        self,
        peptide_hla_pairs: List[Tuple[str, List[str]]],
        threshold: float = 0.5,
        difference: float = 0.1,
        gap: float = 0.15,
        batch_size: int = 64,
    ) -> List[Dict]:
        """
        Optimized (multi-restriction) deconvolution.

        A peptide may be assigned to multiple HLA alleles when their EL
        scores are close.  Three thresholds control the assignment:

        * **T** (threshold) – minimum EL score to be considered.
        * **D** (difference) – if the top score exceeds the second by
          more than D, assign to the top allele only.
        * **G** (gap) – when the top is not dominant, assign to all
          alleles within G of the top score.

        Parameters
        ----------
        peptide_hla_pairs : list of (peptide, [hla1, hla2, ...])
        threshold, difference, gap : float
        batch_size : int

        Returns
        -------
        list of dict with keys ``peptide``, ``assigned_hlas``,
        ``assigned_scores``, ``all_scores``.
        """
        start_time = time.time()
        results: List[Dict] = []

        all_peptides = list({pair[0] for pair in peptide_hla_pairs})
        all_hla_alleles = list(
            {h for _, hlas in peptide_hla_pairs for h in hlas}
        )

        all_el_scores = self._predict_el_scores(
            all_peptides, all_hla_alleles, batch_size
        )

        for peptide, hla_alleles in peptide_hla_pairs:
            el_scores = {
                hla: all_el_scores[peptide][hla] for hla in hla_alleles
            }
            sorted_scores = sorted(
                el_scores.items(), key=lambda x: x[1], reverse=True
            )

            high_score_pairs = [
                (hla, score)
                for hla, score in sorted_scores
                if score > threshold
            ]

            if not high_score_pairs:
                result = {
                    "peptide": peptide,
                    "assigned_hlas": [],
                    "assigned_scores": [],
                    "all_scores": el_scores,
                }
                results.append(result)
                continue

            if len(high_score_pairs) == 1:
                result = {
                    "peptide": peptide,
                    "assigned_hlas": [high_score_pairs[0][0]],
                    "assigned_scores": [high_score_pairs[0][1]],
                    "all_scores": el_scores,
                }
                results.append(result)
                continue

            top_score = high_score_pairs[0][1]
            second_score = high_score_pairs[1][1]

            if top_score - second_score > difference:
                result = {
                    "peptide": peptide,
                    "assigned_hlas": [high_score_pairs[0][0]],
                    "assigned_scores": [high_score_pairs[0][1]],
                    "all_scores": el_scores,
                }
            else:
                top_hlas = []
                top_scores = []
                for hla, score in high_score_pairs:
                    if top_score - score < gap:
                        top_hlas.append(hla)
                        top_scores.append(score)
                    else:
                        break
                result = {
                    "peptide": peptide,
                    "assigned_hlas": top_hlas,
                    "assigned_scores": top_scores,
                    "all_scores": el_scores,
                }

            results.append(result)

        elapsed = time.time() - start_time
        print(
            f"Optimized deconvolution took {elapsed:.1f} sec "
            f"(for {len(peptide_hla_pairs)} peptides)."
        )
        return results


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def deconvolute_ma_data(
    model: ELBaseModel,
    peptides: List[str],
    hla_alleles: List[str],
    method: str = "optimized",
    threshold: float = 0.5,
    difference: float = 0.1,
    gap: float = 0.15,
    batch_size: int = 64,
    cache: Optional[ELScoreCache] = None,
    model_id: Optional[str] = None,
    cache_mode: str = "readwrite",
) -> pd.DataFrame:
    """
    Convenience function: deconvolute MA data and return a DataFrame.

    Parameters
    ----------
    model : ELBaseModel
    peptides : list of str
    hla_alleles : list of str
    method : ``'simple'`` or ``'optimized'``
    threshold : float  (T)
    difference : float  (D, optimized only)
    gap : float  (G, optimized only)
    batch_size : int
    cache, model_id, cache_mode : cache configuration

    Returns
    -------
    pd.DataFrame with columns ``peptide``, ``assigned_hla(s)``,
    ``assigned_score(s)``, ``method``, ``Score_EL``, and optionally
    ``%Rank_EL`` (when *threshold* == 0).
    """
    start_time = time.time()
    deconv = MAELDeconvolution(
        model, cache=cache, model_id=model_id, cache_mode=cache_mode
    )

    peptide_hla_pairs = [(peptide, hla_alleles) for peptide in peptides]

    if method == "simple":
        results = deconv.deconvolute_simple(
            peptide_hla_pairs, threshold, batch_size
        )
        df_data = []
        for result in results:
            df_data.append(
                {
                    "peptide": result["peptide"],
                    "assigned_hla": result["assigned_hla"],
                    "assigned_score": result["assigned_score"],
                    "method": "simple",
                    "Score_EL": result["assigned_score"],
                }
            )
        if threshold == 0:
            scores = [item["Score_EL"] for item in df_data]
            ranks = calculate_percentile_ranks(scores)
            for i, rank in enumerate(ranks):
                df_data[i]["%Rank_EL"] = rank

    elif method == "optimized":
        results = deconv.deconvolute_optimized(
            peptide_hla_pairs, threshold, difference, gap, batch_size
        )
        df_data = []
        for result in results:
            if len(result["assigned_hlas"]) == 0:
                df_data.append(
                    {
                        "peptide": result["peptide"],
                        "assigned_hlas": "None",
                        "assigned_scores": "0.0",
                        "method": "optimized",
                        "Score_EL": 0.0,
                    }
                )
            elif len(result["assigned_hlas"]) == 1:
                assigned_score = result["assigned_scores"][0]
                df_data.append(
                    {
                        "peptide": result["peptide"],
                        "assigned_hlas": result["assigned_hlas"][0],
                        "assigned_scores": assigned_score,
                        "method": "optimized",
                        "Score_EL": assigned_score,
                    }
                )
            else:
                max_score = max(result["assigned_scores"])
                df_data.append(
                    {
                        "peptide": result["peptide"],
                        "assigned_hlas": ";".join(result["assigned_hlas"]),
                        "assigned_scores": ";".join(
                            f"{s}" for s in result["assigned_scores"]
                        ),
                        "method": "optimized",
                        "Score_EL": max_score,
                    }
                )
        if threshold == 0:
            scores = [item["Score_EL"] for item in df_data]
            ranks = calculate_percentile_ranks(scores)
            for i, rank in enumerate(ranks):
                df_data[i]["%Rank_EL"] = rank
    else:
        raise ValueError(f"Unknown method: {method!r}")

    elapsed = time.time() - start_time
    print(f"Total time for {method} deconvolution and rank: {elapsed:.1f} sec.")
    return pd.DataFrame(df_data)


def calculate_percentile_ranks(scores: List[float]) -> List[float]:
    """
    Compute percentile ranks (numpy-vectorised).

    Higher scores → lower percentile (closer to 0);
    lower scores → higher percentile (closer to 100).
    """
    arr = np.array(scores, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        return [0.0] * n
    desc_order = np.argsort(-arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[desc_order] = np.arange(n, dtype=np.float64) / (n - 1) * 100.0
    return ranks.tolist()


def read_fasta_and_predict(
    model: ELBaseModel,
    fasta_file: str,
    hla_alleles: List[str],
    method: str = "optimized",
    threshold: float = 0.5,
    difference: float = 0.1,
    gap: float = 0.15,
    batch_size: int = 64,
    cache: Optional[ELScoreCache] = None,
    model_id: Optional[str] = None,
    cache_mode: str = "readwrite",
) -> pd.DataFrame:
    """
    Read peptides from a FASTA file and run MA deconvolution.

    Returns a DataFrame with an additional ``peptide_id`` column
    preserving the FASTA record identifiers.
    """
    start_time = time.time()

    sequences: List[str] = []
    ids: List[str] = []
    valid_count, total_count = 0, 0
    standard_amino_acids = set(aa_dict_one_hot.keys())

    for record in SeqIO.parse(fasta_file, "fasta"):
        total_count += 1
        seq_str = str(record.seq)
        if all(aa.upper() in standard_amino_acids for aa in seq_str):
            sequences.append(seq_str)
            ids.append(record.id)
            valid_count += 1

    print(f"Peptides before filtering: {total_count}")
    print(f"Peptides filtered out: {total_count - valid_count}")
    print(f"Retention rate: {valid_count / total_count:.2%}")
    print(f"Time for reading FASTA file: {time.time() - start_time:.1f} sec.")

    results_df = deconvolute_ma_data(
        model=model,
        peptides=sequences,
        hla_alleles=hla_alleles,
        method=method,
        threshold=threshold,
        difference=difference,
        gap=gap,
        batch_size=batch_size,
        cache=cache,
        model_id=model_id,
        cache_mode=cache_mode,
    )

    results_df.insert(0, "peptide_id", ids)

    elapsed = time.time() - start_time
    print(f"Total time for reading FASTA and predicting: {elapsed:.1f} sec.")
    return results_df


def filter_by_rank(df: pd.DataFrame, max_rank_pct: float = 2.0) -> pd.DataFrame:
    """
    Filter a DataFrame by ``%Rank_EL``, keeping only rows with rank
    ≤ *max_rank_pct*.
    """
    if "%Rank_EL" not in df.columns:
        raise ValueError("DataFrame does not contain a '%Rank_EL' column.")
    return df[df["%Rank_EL"] <= max_rank_pct].copy()
