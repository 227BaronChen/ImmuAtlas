"""
ELBaseModel – Transformer-based HLA peptide presentation prediction model.

This module contains the core neural network architecture used by ImmuAtlas
for predicting the likelihood that a peptide is presented by a given HLA
molecule (EL – Eluted Ligand score).

Only inference-related code is included.  Training utilities have been
moved to the ``training`` sub-package (if needed).
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from immuatlas.embedding import aa_dict_one_hot, load_hla_library

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

PEPTIDE_MAX_LEN = 25   # max supported peptide length (padded with 'X')
HLA_MAX_LEN = 34       # HLA pseudo-sequence length
PEPTIDE_ENCODING_DIM = 21
HLA_ENCODING_DIM = 21


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class SelfAttentionPooling(nn.Module):
    """
    Self-Attention Pooling layer.

    Learns attention weights to focus on amino-acid residues that contribute
    most to immunogenicity, improving model interpretability.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        sequence_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        sequence_output : Tensor  (batch_size, seq_len, hidden_size)
        mask : Tensor or None     (batch_size, seq_len)

        Returns
        -------
        pooled_output : Tensor    (batch_size, hidden_size)
        attn_weights   : Tensor   (batch_size, seq_len)
        """
        attention_scores = self.attention_weights(sequence_output).squeeze(-1)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attention_scores, dim=1)
        pooled_output = torch.sum(
            sequence_output * attn_weights.unsqueeze(-1), dim=1
        )
        return pooled_output, attn_weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ELBaseModel(nn.Module):
    """
    EL Base Model – Transformer-based immunopeptidome presentation predictor.

    Architecture
    ------------
    1. One-hot encode peptide (≤25 aa) and HLA pseudo-sequence (≤34 aa)
    2. Linear projection to *d_model* dimensions
    3. Concatenate → positional encoding → Transformer encoder
    4. Mean pooling → MLP → sigmoid output (presentation probability)

    Parameters
    ----------
    hla_library_file : str or None
        Path to the HLA library TSV.  When *None*, the bundled library
        shipped with the package is used.
    d_model, nhead, num_encoder_layers, dim_feedforward, dropout
        Standard Transformer hyper-parameters.
    """

    def __init__(
        self,
        hla_library_file: Optional[str] = None,
        d_model: int = 320,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load HLA library
        self.hla_lib: Optional[dict] = None
        if hla_library_file is not None:
            hla_df = pd.read_csv(hla_library_file, sep='\t')
            self.hla_lib = dict(zip(hla_df['Allele Name'], hla_df['MHC pseudo-seq']))

        # Embedding layers
        self.peptide_embedding = nn.Linear(PEPTIDE_ENCODING_DIM, d_model)
        self.hla_embedding = nn.Linear(HLA_ENCODING_DIM, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=100)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Output head
        self.presentation_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_peptide(self, peptide_seq: str) -> torch.Tensor:
        padded = peptide_seq + 'X' * (PEPTIDE_MAX_LEN - len(peptide_seq))
        encoded = [aa_dict_one_hot[aa] for aa in padded]
        return torch.tensor(encoded, dtype=torch.float32)

    def _encode_hla(self, hla_seq: str) -> torch.Tensor:
        encoded = [aa_dict_one_hot[aa] for aa in hla_seq]
        return torch.tensor(encoded, dtype=torch.float32)

    def _encode_sequence_pair(
        self,
        peptide_seq: str,
        hla_allele: Optional[str] = None,
        hla_seq: Optional[str] = None,
    ):
        peptide_encoded = self._encode_peptide(peptide_seq)

        if hla_seq is not None:
            hla_encoded = self._encode_hla(hla_seq)
        elif hla_allele is not None and self.hla_lib is not None:
            seq = self.hla_lib.get(hla_allele)
            if seq is None:
                raise ValueError(f"HLA allele {hla_allele} not found in library")
            hla_encoded = self._encode_hla(seq)
        else:
            raise ValueError(
                "Either hla_seq or hla_allele (with valid hla_lib) must be provided"
            )

        return peptide_encoded, hla_encoded

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        peptide_seq_list=None,
        hla_seq_list=None,
        hla_allele_list=None,
        peptide_tensors=None,
        hla_tensors=None,
    ):
        """
        Predict presentation probabilities.

        Two calling conventions are supported:

        1. **String lists** – pass ``peptide_seq_list`` + one of
           ``hla_seq_list`` / ``hla_allele_list``.
        2. **Pre-encoded tensors** – pass ``peptide_tensors`` and
           ``hla_tensors`` (3-D float tensors, same batch dim).

        Returns
        -------
        Tensor  (batch_size, 1)
            Presentation probabilities in [0, 1].
        """
        if (peptide_tensors is None) != (hla_tensors is None):
            raise ValueError(
                "peptide_tensors and hla_tensors must be provided together"
            )

        device = next(self.parameters()).device

        if peptide_tensors is not None and hla_tensors is not None:
            if peptide_tensors.ndim != 3 or hla_tensors.ndim != 3:
                raise ValueError(
                    "peptide_tensors and hla_tensors must be 3D tensors"
                )
            if peptide_tensors.shape[0] != hla_tensors.shape[0]:
                raise ValueError(
                    "peptide_tensors and hla_tensors must share batch dimension"
                )
            peptide_batch = peptide_tensors.to(device=device, dtype=torch.float32)
            hla_batch = hla_tensors.to(device=device, dtype=torch.float32)
        else:
            if peptide_seq_list is None:
                raise ValueError(
                    "peptide_seq_list must be provided when tensor inputs are not used"
                )

            batch_size = len(peptide_seq_list)
            peptide_tensors_list = []
            hla_tensors_list = []

            for i in range(batch_size):
                if hla_seq_list is not None:
                    p_enc, h_enc = self._encode_sequence_pair(
                        peptide_seq_list[i], hla_seq=hla_seq_list[i]
                    )
                elif hla_allele_list is not None and self.hla_lib is not None:
                    p_enc, h_enc = self._encode_sequence_pair(
                        peptide_seq_list[i], hla_allele=hla_allele_list[i]
                    )
                else:
                    raise ValueError(
                        "Either hla_seq_list or hla_allele_list must be provided"
                    )
                peptide_tensors_list.append(p_enc)
                hla_tensors_list.append(h_enc)

            peptide_batch = torch.stack(peptide_tensors_list).to(device)
            hla_batch = torch.stack(hla_tensors_list).to(device)

        # Embed
        peptide_emb = self.peptide_embedding(peptide_batch)
        hla_emb = self.hla_embedding(hla_batch)

        # Concatenate peptide + HLA
        combined_emb = torch.cat([peptide_emb, hla_emb], dim=1)

        # Positional encoding
        combined_emb = self.pos_encoding(
            combined_emb.transpose(0, 1)
        ).transpose(0, 1)

        # Transformer encoder
        transformer_out = self.transformer_encoder(combined_emb)

        # Global average pooling
        pooled_out = torch.mean(transformer_out, dim=1)

        # Prediction head
        presentation_prob = self.presentation_pred(pooled_out)
        return presentation_prob
