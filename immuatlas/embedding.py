"""
Amino acid encoding dictionaries and HLA library loading for ImmuAtlas.

Provides one-hot encoding for amino acids and utilities to load the HLA
pseudo-sequence library shipped with the package.
"""

import os
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Amino-acid token / one-hot dictionaries
# ---------------------------------------------------------------------------

aa_dict_tokens: Dict[str, int] = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
    'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
    'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20,
}

aa_dict_one_hot: Dict[str, List[int]] = {
    aa: [1 if j == idx else 0 for j in range(len(aa_dict_tokens))]
    for aa, idx in aa_dict_tokens.items()
}

# ---------------------------------------------------------------------------
# HLA library helpers
# ---------------------------------------------------------------------------

def _default_hla_library_path() -> str:
    """Return the path to the bundled HLA library TSV."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'HLA_library3.tsv')


def load_hla_library(path: str | None = None) -> Dict[str, str]:
    """
    Load the HLA pseudo-sequence library.

    Parameters
    ----------
    path : str or None
        Path to the HLA library TSV file.  When *None*, the bundled
        ``data/HLA_library3.tsv`` shipped with the package is used.

    Returns
    -------
    dict
        Mapping from allele name (e.g. ``"A0201"``) to pseudo-sequence string.
    """
    if path is None:
        path = _default_hla_library_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"HLA library file not found: {path}")
    hla_df = pd.read_csv(path, sep='\t')
    return dict(zip(hla_df['Allele Name'], hla_df['MHC pseudo-seq']))
