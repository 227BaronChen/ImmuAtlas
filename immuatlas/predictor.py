"""
Peptide-MHC Presentation Probability Predictor.

Supports pairwise input (list / TSV) and batch file input (FASTA + HLA list).
This is the primary user-facing API of ImmuAtlas.
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import pandas as pd
from Bio import SeqIO
from torch.utils.data import DataLoader

from immuatlas.model import ELBaseModel
from immuatlas.data_loader import PeptideHLADataset2
from immuatlas.embedding import aa_dict_one_hot, load_hla_library

# ---------------------------------------------------------------------------
# Default HLA-I allele list (China top 25)
# ---------------------------------------------------------------------------

DEFAULT_HLA_I = [
    "A1101", "A2402", "A0201", "A3303", "A3001", "A0207", "A0206", "A0101", "A3101",
    "A0301", "A2601", "A0203", "A3201", "A6801", "A2901", "A0210", "A2301", "A6901",
    "A1102", "A0205", "A2420", "A1112", "A1110", "A3009", "A0204",
]

# HLA name mapping (handles format inconsistencies)
HLA_NAME_MAPPING = {
    "HLA-A11:01": "A1101",
    "HLA-A02:01": "A0201",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PeptideMHCPair:
    """肽段-MHC配对"""
    peptide: str
    hla_allele: str
    peptide_id: Optional[str] = None


@dataclass
class PredictionResult:
    """预测结果"""
    peptide_id: str
    peptide: str
    hla_allele: str
    presentation_probability: float


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class PeptideMHCPredictor:
    """Peptide-MHC 呈递概率预测器"""

    def __init__(
        self,
        model_path: str,
        hla_library_file: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初始化预测器

        Parameters
        ----------
        model_path : str
            EL 模型权重文件路径
        hla_library_file : str or None
            HLA 库文件路径。为 None 时使用包内自带的库。
        device : str or None
            计算设备，默认自动选择 (cuda/cpu)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.model_path = model_path
        self.hla_library_file = hla_library_file

        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model()

        # Load HLA library
        self.hla_lib = load_hla_library(hla_library_file)

        # Allowed amino acids
        self.allowed_aas = set(aa_dict_one_hot.keys())

    def _load_model(self) -> ELBaseModel:
        """加载 EL 基础模型"""
        model = ELBaseModel(d_model=320, nhead=8, num_encoder_layers=6, dim_feedforward=512)
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def validate_hlas(self, hla_alleles: Sequence[str]) -> Tuple[List[str], List[str]]:
        """
        验证 HLA 等位基因

        Returns
        -------
        (valid_hlas, invalid_hlas)
        """
        valid_hlas = []
        invalid_hlas = []

        for hla in hla_alleles:
            canonical_hla = HLA_NAME_MAPPING.get(hla, hla)
            if canonical_hla in self.hla_lib:
                valid_hlas.append(canonical_hla)
            else:
                invalid_hlas.append(hla)

        return valid_hlas, invalid_hlas

    def is_valid_peptide(
        self,
        seq: str,
        min_len: int = None,
        max_len: int = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        验证肽段序列

        Returns
        -------
        (is_valid, error_message)
        """
        if not seq:
            return False, "Empty sequence"
        if min_len is not None and len(seq) < min_len:
            return False, f"Sequence too short (min {min_len}): {len(seq)}"
        if max_len is not None and len(seq) > max_len:
            return False, f"Sequence too long (max {max_len}): {len(seq)}"
        invalid_aas = [aa for aa in seq if aa not in self.allowed_aas]
        if invalid_aas:
            return False, f"Invalid amino acids: {set(invalid_aas)}"
        return True, None

    def predict_single_pair(
        self,
        peptide: str,
        hla_allele: str,
        peptide_id: Optional[str] = None,
    ) -> PredictionResult:
        """预测单个肽段-HLA对"""
        is_valid, error_msg = self.is_valid_peptide(peptide)
        if not is_valid:
            raise ValueError(f"Invalid peptide {peptide}: {error_msg}")

        canonical_hla = HLA_NAME_MAPPING.get(hla_allele, hla_allele)
        if canonical_hla not in self.hla_lib:
            raise ValueError(f"Unknown HLA allele: {hla_allele}")

        with torch.no_grad():
            prob = self.model(
                peptide_seq_list=[peptide],
                hla_allele_list=[canonical_hla],
            )
            prob_value = prob.squeeze(-1).item()

        return PredictionResult(
            peptide_id=peptide_id or f"{peptide}_{canonical_hla}",
            peptide=peptide,
            hla_allele=canonical_hla,
            presentation_probability=prob_value,
        )

    def predict_pairs(
        self,
        pairs: Sequence[PeptideMHCPair],
        batch_size: int = 128,
        min_len: int = None,
        max_len: int = None,
    ) -> List[PredictionResult]:
        """批量预测肽段-HLA对"""
        results = []
        valid_pairs = []
        skipped = 0

        for pair in pairs:
            is_valid, error_msg = self.is_valid_peptide(pair.peptide, min_len, max_len)
            if not is_valid:
                print(f"Warning: Skipping {pair.peptide} ({pair.hla_allele}): {error_msg}")
                skipped += 1
                continue

            canonical_hla = HLA_NAME_MAPPING.get(pair.hla_allele, pair.hla_allele)
            if canonical_hla not in self.hla_lib:
                print(f"Warning: Skipping {pair.peptide} ({pair.hla_allele}): Unknown HLA")
                skipped += 1
                continue

            valid_pairs.append((pair, canonical_hla))

        print(f"Processing {len(valid_pairs)} valid pairs (skipped {skipped})")

        with torch.no_grad():
            for i in range(0, len(valid_pairs), batch_size):
                batch_end = min(i + batch_size, len(valid_pairs))
                batch = valid_pairs[i:batch_end]

                peptides = [pair.peptide for pair, _ in batch]
                hla_alleles = [hla for _, hla in batch]

                probs = self.model(
                    peptide_seq_list=peptides,
                    hla_allele_list=hla_alleles,
                )
                prob_values = probs.squeeze(-1).detach().cpu().tolist()
                if not isinstance(prob_values, list):
                    prob_values = [prob_values]

                for (pair, canonical_hla), prob_value in zip(batch, prob_values):
                    results.append(PredictionResult(
                        peptide_id=pair.peptide_id or f"{pair.peptide}_{canonical_hla}",
                        peptide=pair.peptide,
                        hla_allele=canonical_hla,
                        presentation_probability=prob_value,
                    ))

        return results

    def predict_from_tsv(
        self,
        tsv_path: str,
        peptide_col: str = "peptide",
        hla_col: str = "hla",
        peptide_id_col: Optional[str] = None,
        batch_size: int = 128,
        min_len: int = None,
        max_len: int = None,
    ) -> List[PredictionResult]:
        """从 TSV 文件预测"""
        df = pd.read_csv(tsv_path, sep='\t')

        if peptide_col not in df.columns or hla_col not in df.columns:
            raise ValueError(f"Missing columns: {peptide_col}, {hla_col}")

        pairs = []
        for idx, row in df.iterrows():
            peptide_id = row.get(peptide_id_col) if peptide_id_col else None
            pair = PeptideMHCPair(
                peptide=row[peptide_col],
                hla_allele=row[hla_col],
                peptide_id=peptide_id,
            )
            pairs.append(pair)

        return self.predict_pairs(pairs, batch_size, min_len, max_len)

    def predict_from_fasta_and_hlas(
        self,
        fasta_path: str,
        hla_alleles: Sequence[str],
        batch_size: int = 128,
        chunk_size: int = 10000,
        min_len: int = None,
        max_len: int = None,
    ) -> Tuple[List[PredictionResult], Dict[str, int]]:
        """从 FASTA 文件和 HLA 列表批量预测"""
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        valid_hlas, invalid_hlas = self.validate_hlas(hla_alleles)
        if invalid_hlas:
            raise ValueError(f"Unknown HLA alleles: {invalid_hlas}")
        if not valid_hlas:
            raise ValueError("No valid HLA alleles")

        results = []
        stats = {
            'total_records': 0,
            'valid_peptides': 0,
            'skipped_peptides': 0,
            'predictions_made': 0,
        }

        chunk_ids = []
        chunk_peptides = []

        with torch.no_grad():
            for record in SeqIO.parse(fasta_path, "fasta"):
                stats['total_records'] += 1
                seq = str(record.seq).upper()

                is_valid, _ = self.is_valid_peptide(seq, min_len, max_len)
                if not is_valid:
                    stats['skipped_peptides'] += 1
                    continue

                chunk_ids.append(record.id)
                chunk_peptides.append(seq)

                if len(chunk_peptides) >= chunk_size:
                    chunk_results = self._predict_chunk(
                        chunk_ids, chunk_peptides, valid_hlas, batch_size
                    )
                    results.extend(chunk_results)
                    stats['predictions_made'] += len(chunk_results)
                    chunk_ids = []
                    chunk_peptides = []
                    stats['valid_peptides'] += len(chunk_results) // len(valid_hlas)

            if chunk_peptides:
                chunk_results = self._predict_chunk(
                    chunk_ids, chunk_peptides, valid_hlas, batch_size
                )
                results.extend(chunk_results)
                stats['predictions_made'] += len(chunk_results)
                stats['valid_peptides'] += len(chunk_results) // len(valid_hlas)

        return results, stats

    def _predict_chunk(
        self,
        peptide_ids: List[str],
        peptides: List[str],
        hla_alleles: List[str],
        batch_size: int,
    ) -> List[PredictionResult]:
        """预测一个肽段块"""
        results = []

        for hla in hla_alleles:
            hla_seqs = [self.hla_lib[hla]] * len(peptides)
            dataset = PeptideHLADataset2(peptides, hla_seqs)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            hla_scores = []
            for batch_peptides, batch_hla_seqs in dataloader:
                preds = self.model(batch_peptides, hla_seq_list=batch_hla_seqs)
                hla_scores.extend(preds.squeeze(-1).detach().cpu().tolist())

            for pep_id, pep_seq, score in zip(peptide_ids, peptides, hla_scores):
                results.append(PredictionResult(
                    peptide_id=pep_id,
                    peptide=pep_seq,
                    hla_allele=hla,
                    presentation_probability=score,
                ))

        return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_results_to_file(
    results: List[PredictionResult],
    output_prefix: str,
    write_long_format: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    将预测结果写入文件

    Returns
    -------
    (wide_file_path, long_file_path_or_None)
    """
    wide_path = f"{output_prefix}.predictions.wide.tsv"
    long_path = f"{output_prefix}.predictions.long.tsv" if write_long_format else None

    wide_data = {}
    for result in results:
        if result.peptide_id not in wide_data:
            wide_data[result.peptide_id] = {'peptide': result.peptide}
        wide_data[result.peptide_id][result.hla_allele] = f"{result.presentation_probability:.8f}"

    all_hlas = sorted(set(r.hla_allele for r in results))

    with open(wide_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        header = ['peptide_id', 'peptide'] + all_hlas
        writer.writerow(header)
        for pep_id in sorted(wide_data.keys()):
            row = [pep_id, wide_data[pep_id]['peptide']]
            for hla in all_hlas:
                row.append(wide_data[pep_id].get(hla, 'NA'))
            writer.writerow(row)

    print(f"Saved wide format results to {wide_path}")

    if write_long_format:
        with open(long_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['peptide_id', 'peptide', 'hla', 'presentation_probability'])
            for result in results:
                writer.writerow([
                    result.peptide_id,
                    result.peptide,
                    result.hla_allele,
                    f"{result.presentation_probability:.8f}",
                ])
        print(f"Saved long format results to {long_path}")

    return wide_path, long_path


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

class RawTextArgumentDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """自定义帮助格式化器"""

    def _get_help_string(self, action):
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) and action.default in (False, True):
            return help_text
        if action.default is not argparse.SUPPRESS and action.default is not None:
            help_text += " (default: %(default)s)"
        return help_text
