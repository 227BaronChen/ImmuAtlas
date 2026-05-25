"""
ImmuAtlas – Pan-cancer Neoantigen Atlas and AI Screening System
================================================================

A PyTorch-based toolkit for predicting HLA-I peptide presentation
and deconvoluting multiple-allele (MA) mass-spectrometry data.

Key classes
-----------
- :class:`ELBaseModel` – Transformer model for EL score prediction
- :class:`PeptideMHCPredictor` – High-level Python API for single-pair
  and batch prediction
- :class:`MAELDeconvolution` – MA deconvolution (simple & optimized)
- :class:`ELScoreCache` – SQLite-based prediction cache with sharding

Quick start
-----------
>>> from immuatlas import PeptideMHCPredictor
>>> predictor = PeptideMHCPredictor()
>>> result = predictor.predict("LLGATCMV", "HLA-A*02:01")
>>> print(result.score_el)
"""

__version__ = "1.0.0"

from immuatlas.model import ELBaseModel
from immuatlas.predictor import PeptideMHCPredictor, PredictionResult, PeptideMHCPair
from immuatlas.ma_deconvolution import MAELDeconvolution
from immuatlas.cache import ELScoreCache
from immuatlas.embedding import load_hla_library

__all__ = [
    "ELBaseModel",
    "PeptideMHCPredictor",
    "PredictionResult",
    "PeptideMHCPair",
    "MAELDeconvolution",
    "ELScoreCache",
    "load_hla_library",
]
