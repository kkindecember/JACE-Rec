# JACE-Rec

**Joint Adaptive Calibration for Expert-Guided Generative Recommendation**

JACE jointly learns how to combine sequence and semantic experts, allocate
probability to previously interacted items, and guide a frozen generative
recommender. A user-conditioned gate has **18 trainable parameters**. Its
catalog distribution is aggregated over item-ID prefixes to guide generation
while preserving the generator's probability mass on legal tokens.

This repository contains the method, the GRAM training/evaluation interface,
and the JACE decoding adapters for ActionPiece and APAO.

The CPU example runs without external assets. Dataset evaluation uses the
downloadable GRAM backend and one asset archive per dataset. The archives
contain the exact processed inputs and selected model tensors from our runs.
The release archives are prepared locally; their GitHub upload is pending.
Archives are distributed in small parts; the download script joins and verifies them automatically.

## Installation

The GRAM experiments use Python 3.9, PyTorch 1.11.0, Transformers 4.26.0 and
NumPy 1.23.1. From the repository root:

```bash
python -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For a CPU-only installation, replace the first command with
`python -m pip install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu`.

## Quick example

```bash
python examples/demo.py
python -m unittest discover -s tests
```

The example computes the calibrated catalog distribution, verifies the history
mass budget, reconstructs item probabilities from prefix conditionals, and
backpropagates the joint path loss on synthetic data. It is a mechanism example,
not a reproduction of the paper's dataset results.

## Dataset evaluation

Install the pinned backend, then restore the Toys assets from the GitHub
repository hosting the `assets-v1` release:

```bash
python scripts/install_gram_backend.py
python scripts/fetch_assets.py --repository kkindecember/JACE-Rec --domain toys
python scripts/jace.py --config configs/current-toys-2023.json verify-assets --full
python scripts/verify_data.py --config configs/current-toys-2023.json
python scripts/jace.py --config configs/current-toys-2023.json evaluate \
  --methods A6 --output runs/toys-2023-test --device auto
```

Replace `kkindecember/JACE-Rec` with the repository containing the release. For
locally downloaded archives, use `--from /path/to/release-assets` instead.
See [reproduction instructions](docs/REPRODUCE.md) for gate retraining,
upstream training and item encoding.

Configurations cover **Toys, Beauty, Sports and Yelp**, with generator seeds
**2023, 2024 and 2025**. `A6` is JACE; `A0` disables guidance on the same clean
GRAM checkpoint; `A8` uses fixed default coefficients. GPU execution requires
the account policy described in [REPRODUCE.md](docs/REPRODUCE.md).

## Code

| File | Purpose |
|---|---|
| `src/jace_rec/gating.py` | Features, 18-parameter gate, catalog probabilities and path loss |
| `src/jace_rec/experts.py` | Frozen SASRec and full-history semantic scores |
| `src/jace_rec/tree.py`, `decoding.py` | Prefix aggregation and probability guidance |
| `src/jace_rec/training.py`, `evaluation.py` | Gate fitting, internal calibration and evaluation |
| `src/jace_rec/runtime.py`, `upstream.py` | GRAM integration and clean upstream training |
| `src/jace_rec/adapters/` | ActionPiece and APAO decoding adaptations |

## License and acknowledgments

Original JACE code is licensed under [Apache-2.0](LICENSE). The sequence expert
builds on LIME-Rec. GRAM, ActionPiece, APAO, T5 and Qwen retain their own source,
model and data terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
