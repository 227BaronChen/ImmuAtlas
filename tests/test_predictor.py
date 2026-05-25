"""
Tests for ImmuAtlas – model, embedding, predictor, and deconvolution.

Run from repo root:
    python -m pytest tests/ -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from immuatlas.model import ELBaseModel, PEPTIDE_MAX_LEN, HLA_MAX_LEN
from immuatlas.embedding import aa_dict_one_hot, aa_dict_tokens, load_hla_library
from immuatlas.data_loader import PeptideHLADataset2
from immuatlas.ma_deconvolution import (
    MAELDeconvolution,
    calculate_percentile_ranks,
    filter_by_rank,
    deconvolute_ma_data,
)
from immuatlas.utils import format_params_count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Create a small ELBaseModel for testing (random weights)."""
    return ELBaseModel(
        d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128
    )


@pytest.fixture
def hla_lib():
    """Load the bundled HLA library."""
    return load_hla_library()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestELBaseModel:
    def test_forward_shape(self, model):
        """Model output should be (batch_size, 1)."""
        batch = 4
        pep_tensor = torch.randn(batch, PEPTIDE_MAX_LEN, 21)
        hla_tensor = torch.randn(batch, HLA_MAX_LEN, 21)
        with torch.no_grad():
            out = model(peptide_tensors=pep_tensor, hla_tensors=hla_tensor)
        assert out.shape == (batch, 1)

    def test_output_range(self, model):
        """Output should be in [0, 1] (sigmoid)."""
        batch = 8
        pep_tensor = torch.randn(batch, PEPTIDE_MAX_LEN, 21)
        hla_tensor = torch.randn(batch, HLA_MAX_LEN, 21)
        with torch.no_grad():
            out = model(peptide_tensors=pep_tensor, hla_tensors=hla_tensor)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_aa_dict_tokens_completeness(self):
        """aa_dict_tokens should cover 20 standard AAs + X."""
        assert len(aa_dict_tokens) == 21
        assert "X" in aa_dict_tokens or 20 in aa_dict_tokens.values()

    def test_aa_dict_one_hot_completeness(self):
        """aa_dict_one_hot should cover 20 standard AAs."""
        assert len(aa_dict_one_hot) == 20

    def test_load_hla_library(self, hla_lib):
        """HLA library should load and contain common alleles."""
        assert len(hla_lib) > 1000
        # Common alleles should be present (with or without HLA- prefix)
        assert any("A0201" in k for k in hla_lib)


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_dataset2_creation(self):
        """PeptideHLADataset2 should create correctly."""
        peptides = ["LLGATCMV", "GILGFVFTL"]
        hla_seqs = ["YVDGY", "YVDGY"]
        dataset = PeptideHLADataset2(peptides, hla_seqs)
        assert len(dataset) == 2
        pep, hla = dataset[0]
        assert pep.shape == (PEPTIDE_MAX_LEN, 21)
        assert hla.shape[2] == 21


# ---------------------------------------------------------------------------
# MA Deconvolution tests
# ---------------------------------------------------------------------------


class TestMADeconvolution:
    def test_batch_encode_peptides(self):
        """_batch_encode_peptides should produce correct shape."""
        peptides = ["LLGATCMV", "GILGFVFTL"]
        tensor = MAELDeconvolution._batch_encode_peptides(peptides)
        assert tensor.shape == (2, PEPTIDE_MAX_LEN, 21)

    def test_batch_encode_hlas(self):
        """_batch_encode_hlas should produce correct shape."""
        seqs = ["YVDGY", "YVDGY"]
        tensor = MAELDeconvolution._batch_encode_hlas(seqs)
        assert tensor.shape == (2, 5, 21)

    def test_deconvolute_simple(self, model, hla_lib):
        """Simple deconvolution should return results for each peptide."""
        deconv = MAELDeconvolution(model)
        pairs = [
            ("LLGATCMV", ["HLA-A*02:01", "HLA-B*07:02"]),
            ("GILGFVFTL", ["HLA-A*02:01", "HLA-B*07:02"]),
        ]
        results = deconv.deconvolute_simple(pairs, threshold=0.0)
        assert len(results) == 2
        for r in results:
            assert "peptide" in r
            assert "assigned_hla" in r
            assert "assigned_score" in r
            assert "all_scores" in r

    def test_deconvolute_optimized(self, model, hla_lib):
        """Optimized deconvolution should return results for each peptide."""
        deconv = MAELDeconvolution(model)
        pairs = [
            ("LLGATCMV", ["HLA-A*02:01", "HLA-B*07:02"]),
            ("GILGFVFTL", ["HLA-A*02:01", "HLA-B*07:02"]),
        ]
        results = deconv.deconvolute_optimized(pairs, threshold=0.0)
        assert len(results) == 2
        for r in results:
            assert "assigned_hlas" in r
            assert "assigned_scores" in r


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_percentile_ranks(self):
        """Percentile ranks should be in [0, 100] and ordered."""
        scores = [0.9, 0.5, 0.7, 0.3, 0.1]
        ranks = calculate_percentile_ranks(scores)
        assert len(ranks) == 5
        assert all(0 <= r <= 100 for r in ranks)
        # Highest score should have lowest rank
        assert ranks[0] == 0.0

    def test_filter_by_rank(self):
        """filter_by_rank should correctly filter DataFrame."""
        df = pd.DataFrame({
            "peptide": ["A", "B", "C"],
            "%Rank_EL": [1.0, 5.0, 10.0],
        })
        filtered = filter_by_rank(df, max_rank_pct=5.0)
        assert len(filtered) == 2

    def test_filter_by_rank_missing_column(self):
        """filter_by_rank should raise if %Rank_EL is missing."""
        df = pd.DataFrame({"peptide": ["A"]})
        with pytest.raises(ValueError):
            filter_by_rank(df)

    def test_format_params_count(self):
        """format_params_count should return a string."""
        model = ELBaseModel(d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128)
        result = format_params_count(model)
        assert isinstance(result, str)
        assert "M" in result or "K" in result


# ---------------------------------------------------------------------------
# Integration test (requires model weights – skip if not available)
# ---------------------------------------------------------------------------


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "models",
    "n10_100pct.best_N6_BCELoss.bs64_lr1e-4.epoch16.pth"
)


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Pre-trained model weights not found"
)
class TestPredictorIntegration:
    def test_predictor_single(self):
        """PeptideMHCPredictor should produce a valid score."""
        from immuatlas import PeptideMHCPredictor
        predictor = PeptideMHCPredictor(model_path=MODEL_PATH)
        result = predictor.predict("LLGATCMV", "HLA-A*02:01")
        assert 0.0 <= result.score_el <= 1.0

    def test_predictor_batch(self):
        """PeptideMHCPredictor batch should return correct count."""
        from immuatlas import PeptideMHCPredictor
        predictor = PeptideMHCPredictor(model_path=MODEL_PATH)
        pairs = [
            ("LLGATCMV", "HLA-A*02:01"),
            ("GILGFVFTL", "HLA-A*02:01"),
        ]
        results = predictor.predict_batch(pairs)
        assert len(results) == 2
