# Reproducing JACE

## Release scope

The method and CPU checks are self-contained. Dataset runs additionally install
the pinned GRAM backend and restore the dataset asset archives. The four archives are available under the `assets-v1` GitHub release tag:

```bash
python scripts/install_gram_backend.py
python scripts/fetch_assets.py --repository kkindecember/JACE-Rec --domain toys
```

Use `--domain all` for all four datasets. Archives are distributed as 64 MiB
parts, which the script downloads, verifies and joins automatically. A failed
download resumes from cached parts on the next invocation. Each archive includes all three seeds,
precomputed Qwen vectors, split definitions and the inputs for upstream/gate
retraining. For archives already on disk, replace `--repository kkindecember/JACE-Rec`
with `--from /path/to/release-assets`. T5 files are downloaded directly from the
pinned Hugging Face revision; `--t5-source /path/to/t5-small` supports an offline
copy. Downloads and extraction verify SHA256 and do not replace different
existing assets. Interrupted HTTP downloads retain resumable `.part` files.

`assets/manifest.json` records the exact sizes and SHA256 hashes of 89 files
needed for saved-weight GRAM/JACE evaluation and gate-cache reconstruction.
Paths are relative to this repository. Files shared across configurations are
listed once. The manifest is an integrity record, not an asset download or a
redistribution license.

`assets/training-manifest.json` adds the clean upstream sequences, training
prefixes, source text/sequences and fixed encoder inputs. `assets/distribution.json`
records archive names, contents, sizes and hashes. Exported checkpoint tensors
are identical to the selected research models; optimizer/RNG state and internal
path metadata are omitted. File hashes therefore identify the release files.

The backend installer downloads GRAM commit
`7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`, checks every source hash, removes
unused IPython debug imports and replaces wrapper `view` operations with
`reshape`. It installs into the ignored `vendor/gram/` directory. An offline
upstream source copy is accepted with `--from /path/to/GRAM/src`.
The exact transformations and hashes are in `assets/gram-backend-manifest.json`.

The research snapshot also contained inactive cross-attention-cache controls
and extra collator metadata. They are not needed by the released computation.
Cross-process CPU checks on every dataset verified identical expert scores,
encoder outputs, teacher logits, labeled forward loss, and A0/A6/A8 top-50
predictions/scores (two forward examples and one decoded user per dataset).
This validates the backend substitution on those checks, not a new full test run.

## Data and checkpoints

Restore files to the paths specified in the manifest, then run `verify-assets
--full` for the chosen configuration. The main layout is:

```text
assets/
  data/{Domain}/
    catalog.json
    train_samples.json
    calibration_samples.json
    validation_samples.json
    test_samples.json
    split.json
    qwen4_embeddings.npz
    upstream_sequences.jsonl
    gram_train.jsonl
    encoder_texts.json
    raw/
  checkpoints/{Domain}/{seed}/
    gram.pt
    sasrec.pt
    head.pt
    normalization.json
    catalog.json                 # when specified by the configuration
  encoders/t5-small/
```

The configured catalog must match the generator and sequence-expert seed:
its collaborative passages are not interchangeable across runs.

Each sample has `user_id`, `full_history` (oldest to newest), `history` (at most
the last 20 items), and `target`. Item indices are 1-based. The catalog contains
`items`, `lexical_ids`, `passages` (each with an index-zero placeholder), and
`decoder_paths` (one path per real item). Paths start with token 0 and end with
EOS token 1. The embedding archive contains `item_ids` in `catalog.items[1:]`
order and an `embeddings` float32 array. `split.json` identifies `fit_indices`
and `calibration_indices` within `train_samples.json`.

The starting data are GRAM's preprocessed sequences and lexical IDs. The last
event is the test target and the penultimate event is the validation target.
For the preselected holdout cohort, remove the terminal event from the remaining
training prefix **before** training clean GRAM and SASRec. All clean prefixes
are retained for GRAM training, with at most 20 history items per prefix.
The frozen gate pool is a subset of those held-out events; preserve the supplied
pool and fit/calibration indices instead of resampling them.

| Dataset | Users | Held-out training events | Gate fit | Internal calibration | Clean GRAM prefixes |
|---|---:|---:|---:|---:|---:|
| Toys | 19,412 | 3,494 | 2,218 | 555 | 105,867 |
| Beauty | 22,363 | 4,085 | 2,648 | 662 | 127,328 |
| Sports | 35,598 | 6,678 | 4,287 | 1,072 | 182,865 |
| Yelp | 30,431 | 5,694 | 3,648 | 913 | 219,367 |

`scripts/verify_data.py` checks these boundaries against the source sequences,
including every clean training prefix. Feature normalization uses fit users
only. Checkpoint selection uses internal calibration, then test is evaluated
with the selection frozen. Repeated Yelp events are retained, including earlier
occurrences of a held-out target item. This is the experimental processed-data
route; rebuilding raw review filtering or upstream lexical-ID training is not
required to use the released inputs.

## GPU admission

All CLI GPU jobs use `scripts/gpu_account_policy.py`. Admission checks account
CUDA contexts and pending leases under a shared file lock, subtracts reserved
memory, prefers already occupied GPUs, and registers the worker before
releasing the lock. Query failures postpone admission.

On a shared host, use the same policy root as the other account workloads:

```bash
export JACE_GPU_POLICY_ROOT=/path/to/shared/project
```

That root contains `configs/gpu-account-policy.json` and
`status/checkpoint_gpu_leases/admission.lock`. A containing project policy is
automatically reused when this checkout is nested inside another project.
For a new standalone installation without an existing policy, set the limit
explicitly, for example:

```bash
python scripts/configure_gpu.py --max-gpus 1
```

The numeric limit is read from the live configuration. Default reservation is
22000 MiB per CLI worker; set the top-level `--memory-mib` for your hardware.
`--device cpu` also works, but full GRAM evaluation is slow on CPU.

## Retrain the gate

With frozen upstream assets installed, run from the repository root:

```bash
python scripts/jace.py --config configs/current-toys-2023.json cache-training \
  --output runs/toys-cache --write-config configs/toys-cache.local.json --device auto
python scripts/jace.py --config configs/toys-cache.local.json train-gate \
  --output runs/toys-heads
python scripts/jace.py --config configs/toys-cache.local.json calibrate \
  --head-dir runs/toys-heads --output runs/toys-calibration --device auto
python scripts/jace.py --config configs/toys-cache.local.json select \
  --calibration runs/toys-calibration --head-dir runs/toys-heads \
  --output runs/toys-selection.json
python scripts/jace.py --config configs/toys-cache.local.json evaluate \
  --head-dir runs/toys-heads --selection runs/toys-selection.json \
  --methods A0 A6 A8 --output runs/toys-test --device auto
```

The gate uses float64, AdamW, learning rate 0.01, weight decay 0.001, batch size
32, and 400 updates. Candidate snapshots are steps 0, 300 and 400. Select by
complete internal-calibration NDCG@10, then Recall@10, then the earlier step.
Selecting step 0 is allowed. Generator seeds 2023/2024/2025 pair with expert
and gate seeds 0/1/2. Outputs retain the implementation name `joint_user_alpha`
for A6.

Use a new output directory after changing inputs, configuration or code.
Each run takes a verified code snapshot; evaluation can resume only when its
input and model identities match. Reported metrics use the full catalog with
previously interacted items retained as candidates.

## Upstream training

`train-sasrec` and `train-gram` support fresh training with `--write-config` to
propagate the resulting asset paths. The dataset archives include
`upstream_sequences.jsonl` (clean `user_id`/`history` rows),
`raw/item_plain_text.txt` (item ID followed by text), and `gram_train.jsonl`
(records with `user_id`, `history` and `target` fields).
Their hashes are in the separate training manifest.

Train the sequence expert first, then use its generated catalog to train GRAM,
then build a new gate cache. Keep the split, lexical IDs and item ordering fixed.
The supplied checkpoint configurations reproduce saved models. New upstream
training creates a fresh experiment; cross-device bitwise retraining equivalence
is not assumed.

Semantic inference uses precomputed Qwen3-Embedding-4B item vectors. It pools
all visible historical item vectors with position decay 0.1, followed by
cosine similarity and full-catalog min-max scaling. It does not remove a
candidate's own historical occurrences. The sequence expert uses the last
50 interactions; GRAM uses the last 20.

To regenerate item vectors, use a **separate Python 3.11 environment** with
`requirements-encoder.txt`. The saved `encoder_texts.json` contains the exact
text inputs used in the experiments, after the established BGE prefix budget.
The encoder is Qwen3-Embedding-4B revision
`5cf2132abc99cad020ac570b19d031efec650f2b`, with no instruction prefix, left
padding, last actual token pooling, float16 model weights and L2-normalized
float32 output. Download it and encode into a new directory:

```bash
python -m pip install -r requirements-encoder.txt
python scripts/fetch_assets.py --encoder
python scripts/jace.py --config configs/current-toys-2023.json encode-items \
  --output runs/toys-encoding --write-config configs/toys-encoding.local.json --device auto
```

This command uses the same shared GPU admission policy. Saved vectors are
already included in the dataset archive, so encoding is optional for evaluation.

## Other generators

`adapters/actionpiece.py` exposes `canonical_paths`, `history_batch` and
`NativePredictor`. Build an `ExpertTrie` from canonical paths, compute the
catalog distribution with `gating.catalog_scores`, and obtain prefix
conditionals with `tree.conditionals`. Pass one set of conditionals and one
alpha per user to `NativePredictor`. This retains ActionPiece's native
noncanonical-token probabilities, five generated tokens, five history
segmentations and native rank fusion.

`adapters/apao.py` exposes `GroupedIndex`, `history_batch`, `decode` and
`teacher_logits`. Build `GroupedIndex` from four-token SIDs prefixed with 0;
call `log_masses` with item log probabilities and pass the masses and alphas
to `decode`. All items sharing a SID contribute to its mass. Resolve output
collisions using `representative`, which selects the smallest source item ID.
The adapter generates four tokens with beam 20.

These are method adapters, not end-to-end baseline launchers. Load and freeze
the appropriate upstream model in its own environment before using them.
The GRAM environment is not the ActionPiece/APAO environment. Adapter tensor
tests do not certify a full upstream training run. Each dataset, seed and
generator fits its own 18-parameter gate.
