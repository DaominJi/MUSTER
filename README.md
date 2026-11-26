# MUSTER

Code, data manifests, and technical report for **MUSTER: From Self-Supervised Expertise Matching to Certified Submodular Reviewer Panel Expansion**.

## Technical Report

The full technical report, including complete proofs, implementation details, per-benchmark breakdowns, and extended panel-expansion experiments, is included in this repository:

- [MUSTER_Technical_Report.pdf](MUSTER_Technical_Report.pdf)

## Repository Structure

| Path | Description |
|---|---|
| `MUSTER_Technical_Report.pdf` | Full technical report with appendices and proofs |
| `muster/` | Dual-branch matcher, negative screen, curriculum, training loop, validation, benchmark loaders, and metrics |
| `expansion/` | Panel expansion: exact assignment oracle with `InsertReviewer`, `CertGreedy` and its greedy variants, baselines, exact optimum, and the three recruitment scenarios ([docs/EXPANSION.md](docs/EXPANSION.md)) |
| `scripts/` | CLI entry points for evaluation, validation, training, encoding, data construction, and panel expansion |
| `checkpoints/` | MD5 manifest for trained dual-branch heads (`muster_sp2.pt`, `muster_sci.pt`) |
| `data/` | MD5 manifest, work indices, fitted negative-screen models, and data documentation ([docs/DATA.md](docs/DATA.md)) |
| `benchmarks/` | MD5 manifest, encoding paper IDs, and benchmark documentation ([benchmarks/README.md](benchmarks/README.md)) |
| `docs/` | Corpus provenance and preprocessing specification, and the panel-expansion guide |
| `tests/` | Unit and data-consistency tests |

The repository holds the expertise-matching configuration reported in the paper: one dual-branch head on frozen SPECTER2 embeddings, one on frozen SciNCL embeddings, and the default dual-backbone model averaging both heads (`w_sci = 0.50`). It also holds the panel-expansion algorithms of the paper, which take the matcher's affinities as input.

## Setup

Python 3.10 or newer with PyTorch, NumPy, and SciPy:

```bash
pip install -r requirements.txt
```

`transformers` is only required when encoding new papers via `scripts/encode.py`. Evaluation and validation run on CPU, whereas training requires a CUDA GPU. The panel-expansion code runs on CPU and additionally needs `numba` and `ortools`.

Large binary assets (trained checkpoints, precomputed frozen embeddings, corpus archive, and episode arrays) are excluded from Git and distributed as release archives in the GitHub Releases page (`v1.0`). Extract any downloaded archive at the repository root and verify integrity against the `MD5SUMS` file in `checkpoints/`, `data/`, or `benchmarks/`.

## Usage

### Evaluate on the Seven Benchmarks

```bash
python scripts/evaluate.py --out runs/eval
```

This scores the three preference benchmarks (`LR-PC`, `LR-RC`, `CMU Gold`) and four conference ranking benchmarks (`NIPS`, `SIGIR`, `SciRepEval`, `KDD`) with `MUSTER (SPECTER2)`, `MUSTER (SciNCL)`, and `MUSTER (Default)`. Pass custom checkpoints via `--sp2` and `--sci`.

### Validate on Held-Out Authors

```bash
python scripts/validate.py --encoder sp2
```

Computes benchmark-free validation metrics over the 2,000 held-out validation authors, including the validation-suite composite MRR and long-history MRR.

### Train a Dual-Branch Head

```bash
python scripts/train.py --encoder sp2 --out runs/sp2_s42
```

Trains one 44.89M-parameter head with AdamW and exponential moving average weights over 90,000 temporal authorship episodes (4,218 steps at batch size 128), saves the best EMA checkpoint to `runs/sp2_s42/head.pt`, and evaluates it on the held-out validation split.

### Rebuild Training Data and Encodings

```bash
python scripts/build_training_data.py --check
python scripts/build_references.py --check
python scripts/fit_screen.py --encoder sp2 --check
python scripts/encode.py check --encoder sp2
```

- `build_training_data.py`: builds author records, training episodes, and validation episodes from the decontaminated OpenAlex corpus.
- `build_references.py`: builds the sparse citation reference index used by the offline negative screen.
- `fit_screen.py`: fits the cross-fitted two-class MAP EM qualification screen.
- `encode.py`: encodes corpus works (`corpus`), benchmark papers (`benchmarks`), or verifies a random sample (`check`) with the frozen scientific encoders.

### Expand a Reviewer Panel

```bash
python scripts/make_synthetic_instance.py runs/toy_instance --tag toy
NUMBA_NUM_THREADS=1 python scripts/expand.py runs/toy_instance toy topic_gap
python scripts/expand_exact.py runs/toy_instance toy emergency
```

- `make_synthetic_instance.py`: writes a small synthetic venue instance in the format of the panel-expansion experiments.
- `expand.py`: runs `CertGreedy`, its exact-marginal greedy variants (`Lazy Greedy`, `Incremental-Flow Greedy`, `Naive Greedy`), `Stochastic Greedy`, and `Top-k retrieval` in the `initialization`, `topic_gap`, or `emergency` scenario, and prints gain, exact marginal evaluations, and selection time per budget.
- `expand_exact.py`: computes the exact optimum per budget with HiGHS.
- `expand_subinstances.py`: compares `CertGreedy` with the exact optimum on small sub-instances.

The instance format, the scenarios, and the timing protocol are described in [docs/EXPANSION.md](docs/EXPANSION.md).

### Run Tests

```bash
python -m pytest tests
```

## Model Overview

A submission and a reviewer's publication history are encoded once by a frozen scientific encoder (`allenai/specter2_base` or `malteos/scincl`). A lightweight adapter applies learnable corpus-mean centering and a zero-initialized residual MLP. The core branch selects a compact support set of recent and representative historical papers, contextualizes them with four gated set attention blocks, and summarizes them into up to 12 research-line prototypes scored via a mass-weighted smooth maximum. The niche branch attends directly from the submission over the full publication history using distinctiveness-adjusted keys that subtract a learned fraction of the profile centroid. An adaptive gating network fuses both branches alongside a calibration offset over history length and topical coherence.

Self-supervised training constructs temporal authorship episodes from peer-reviewed computer science venues, filters candidate in-batch negatives using cross-fitted multi-view MAP EM over authorship, semantic proximity, and citation overlap, and optimizes a bidirectional ranking loss under a confidence-aware Bernstein easy-to-hard curriculum regularized by listwise distillation and prototype coverage and diversity terms. All default hyperparameters are defined in `muster/config.py`.

Panel expansion takes the matcher's affinities, the review quotas of the submissions, and the workload bounds of the reviewers, and selects up to `k` external candidates that maximize the value of the optimal assignment of the enlarged panel. `CertGreedy` computes every marginal gain exactly by inserting the candidate into the current optimal min-cost flow (`InsertReviewer`), and it skips the candidates whose upper bound cannot win the round, using dual bounds from the shadow prices of the current assignment and lazy bounds carried over from earlier rounds. The pruning is certified: the selected panel equals that of the unpruned greedy algorithm.

## Data and Licensing Notes

The OpenAlex training corpus is released under CC0 (see [docs/DATA.md](docs/DATA.md)). Benchmark datasets remain governed by their original authors' licenses and terms (see [benchmarks/README.md](benchmarks/README.md)).
