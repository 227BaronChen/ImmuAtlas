# ImmuAtlas

**Pan-cancer Neoantigen Atlas and AI Screening System**

A PyTorch-based toolkit for predicting HLA-I/II peptide presentation scores (EL, Expression Level) and deconvoluting multiple-allele (MA) mass-spectrometry data.

---

## Features

- **EL Score Prediction** – Transformer-based model predicts the likelihood that a peptide is presented by a given HLA allele.
- **MA Deconvolution** – Assigns peptides to their most likely presenting HLA allele(s) from multi-allele MS data, using either a simple winner-take-all or an optimized multi-restriction method.
- **SQLite Cache** – Sharded, quantised prediction cache with WAL mode for fast incremental scoring.
- **Batch Processing** – FASTA-based batch prediction with checkpointing and incremental cache support.
- **Python API & CLI** – Use as a library or run from the command line.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/227BaronChen/ImmuAtlas.git
cd ImmuAtlas
```

### 2. Create a conda environment (recommended)

```bash
conda create -n immuatlas python=3.10 -y
conda activate immuatlas
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package (optional, for `import immuatlas`)

```bash
pip install -e .
```

### 5. Download model weights

Pre-trained weights are **not** included in the repository due to file size. Place the model checkpoint under `data/models/`:

```
data/models/n10_100pct.best_N6_BCELoss.bs64_lr1e-4.epoch16.pth
```

> The HLA pseudo-sequence library (`data/HLA_library3.tsv`, ~1 MB) **is** included.

---

## Quick Start

### Python API

```python
from immuatlas import PeptideMHCPredictor

# Initialize predictor (loads model + HLA library automatically)
predictor = PeptideMHCPredictor()

# Single prediction
result = predictor.predict("LLGATCMV", "HLA-A*02:01")
print(f"Score_EL = {result.score_el:.4f}")

# Batch prediction
pairs = [
    ("LLGATCMV", "HLA-A*02:01"),
    ("KVLEYVIK", "HLA-B*07:02"),
]
results = predictor.predict_batch(pairs)
for r in results:
    print(f"{r.peptide} × {r.hla} → {r.score_el:.4f}")
```

### MA Deconvolution (Python)

```python
from immuatlas import MAELDeconvolution, PeptideMHCPredictor

predictor = PeptideMHCPredictor()
deconv = MAELDeconvolution(predictor.model)

peptide_hla_pairs = [
    ("LLGATCMV", ["HLA-A*02:01", "HLA-B*07:02"]),
    ("KVLEYVIK", ["HLA-A*02:01", "HLA-B*07:02"]),
]

# Simple method (winner-take-all)
results = deconv.deconvolute_simple(peptide_hla_pairs, threshold=0.5)

# Optimized method (multi-restriction)
results = deconv.deconvolute_optimized(peptide_hla_pairs, threshold=0.5, difference=0.1, gap=0.15)
```

### Command Line – Batch EL Prediction

```bash
python scripts/batch_el_pred.py \
  --fasta-file examples/example_peptides.fasta \
  --hla-alleles A0201 B0702 C0702 \
  --batch-size 128 \
  --output-prefix results/example
```

### Command Line – MA Deconvolution

```bash
python scripts/ma_deconvolute.py \
  --fasta-file examples/example_peptides.fasta \
  --hla-alleles A0201,B0702,C0702 \
  --method optimized \
  --threshold 0.0 \
  --top-pct 3,10,30 \
  --output-prefix results/example
```

---

## Project Structure

```
ImmuAtlas/
├── immuatlas/                  # Core package
│   ├── __init__.py
│   ├── model.py                # ELBaseModel (Transformer)
│   ├── embedding.py            # Amino acid encoding & HLA library
│   ├── data_loader.py          # PyTorch Dataset classes
│   ├── predictor.py            # PeptideMHCPredictor (Python API)
│   ├── batch_predict.py        # Batch prediction core logic
│   ├── ma_deconvolution.py     # MA deconvolution
│   ├── cache.py                # SQLite prediction cache
│   └── utils.py                # Model parameter utilities
├── scripts/                    # CLI entry points
│   ├── batch_el_pred.py        # Batch EL prediction CLI
│   └── ma_deconvolute.py       # MA deconvolution CLI
├── data/                       # Bundled data
│   ├── HLA_library3.tsv        # ~16,000 HLA alleles + pseudo-sequences
│   └── models/                 # Pre-trained weights (not in Git)
├── examples/                   # Example data & scripts
│   ├── example_peptides.fasta
│   ├── example_pairs.tsv
│   └── quick_start.py
├── tests/
│   └── test_predictor.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

---

## Model Architecture

| Parameter            | Value |
|----------------------|-------|
| `d_model`            | 320   |
| `nhead`              | 8     |
| `num_encoder_layers` | 6     |
| `dim_feedforward`    | 512   |
| `dropout`            | 0.1   |
| Input (peptide)      | 25 × 21 one-hot → linear |
| Input (HLA)          | 34 × 21 one-hot → linear |
| Output               | sigmoid (0–1 EL score) |

---

## HLA Allele Naming

The model accepts both compact (`A0201`) and full (`HLA-A*02:01`) formats. Common aliases are automatically normalised via an internal mapping table.

---

## Caching

Enable the SQLite cache to avoid redundant predictions:

```bash
python scripts/batch_el_pred.py \
  --fasta-file input.fasta \
  --hla-alleles A0201 \
  --cache-db el_cache.sqlite \
  --cache-mode readwrite \
  --incremental
```

Cache features:
- **Sharding** – auto-adaptive or fixed shard count
- **Quantisation** – uint16 (default) or float32 score storage
- **Incremental mode** – only predict peptides missing from cache
- **Import** – bulk-import existing TSV results into cache

---

## License

This project is licensed under the [MIT License](LICENSE).
