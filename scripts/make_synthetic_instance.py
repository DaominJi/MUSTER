#!/usr/bin/env python3
"""Write a small synthetic venue instance in the format that expansion/scenarios.py reads (docs/EXPANSION.md), so that
every panel-expansion script and test runs without the expert repository behind the paper's experiments.

Reviewers and submissions are noisy mixtures of topics, and the affinity of a pair grows with the cosine of their topic
vectors. A share of the reviewers are the venue's earlier authors (the incumbent pool). The venue's earlier papers follow
an older topic distribution, so some topics have grown.
usage: python scripts/make_synthetic_instance.py OUT_DIR [--tag toy] [--submissions 120] [--reviewers 3000] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCALE = 1_000_000  # micro-units per unit of affinity (expansion/flow.py)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make(out_dir, tag="toy", n_sub=120, n_rev=3000, seed=0, dim=64, n_topics=10, pool_share=0.3, conflict_share=0.01):
    rng = np.random.default_rng(seed)
    centers = unit(rng.normal(size=(n_topics, dim)))
    p_now, p_past = rng.dirichlet(np.full(n_topics, 2.0)), rng.dirichlet(np.full(n_topics, 2.0))

    def papers(n, p):
        return unit(centers[rng.choice(n_topics, size=n, p=p)] + 0.6 * rng.normal(size=(n, dim)) / np.sqrt(dim))

    sub, past = papers(n_sub, p_now), papers(4 * n_sub, p_past)
    second = rng.random(n_rev) < 0.5  # half of the reviewers work on two topics
    rev = unit(
        centers[rng.integers(n_topics, size=n_rev)]
        + 0.6 * second[:, None] * centers[rng.integers(n_topics, size=n_rev)]
        + 0.6 * rng.normal(size=(n_rev, dim)) / np.sqrt(dim)
    )
    aff = np.clip(0.5 + 0.5 * (rev @ sub.T) + 0.03 * rng.normal(size=(n_rev, n_sub)), 0.0, 1.0)
    A = np.rint(aff * SCALE).astype(np.int32)
    A[rng.random(A.shape) < conflict_share] = -1  # conflicts of interest
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{tag}_A.npy", A)
    np.savez(out_dir / f"{tag}_spec.npz", inc_pool=rng.random(n_rev) < pool_share)
    np.savez(out_dir / f"topic_{tag}.npz", sub=sub.astype(np.float32), past=past.astype(np.float32))
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tag", default="toy")
    ap.add_argument("--submissions", type=int, default=120)
    ap.add_argument("--reviewers", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.out_dir, a.tag, a.submissions, a.reviewers, a.seed)
    print(
        f"wrote {a.tag}_A.npy, {a.tag}_spec.npz and topic_{a.tag}.npz to {a.out_dir} "
        f"({a.reviewers} reviewers, {a.submissions} submissions)"
    )


if __name__ == "__main__":
    main()
