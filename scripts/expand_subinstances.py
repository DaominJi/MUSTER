#!/usr/bin/env python3
"""CertGreedy against the exact optimum on random sub-instances of a venue (100 submissions, 500 candidates).

usage: python scripts/expand_subinstances.py INSTANCE_DIR TAG {initialization,topic_gap,emergency} [--reps 10] [--limit 600]

Each sub-instance is small enough for the mixed-integer program over all candidates (HiGHS, strong formulation). When a
MIP does not close within the time limit, its best bound is kept, and the printed share of CertGreedy is a lower bound.
Top-k retrieval and Stochastic Greedy (mean of five runs) are evaluated on the same sub-instances.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expansion import flow as fl  # noqa: E402
from expansion.baselines import stochastic_greedy, topk_retrieval  # noqa: E402
from expansion.certgreedy import greedy  # noqa: E402
from expansion.exact import mip_optimum  # noqa: E402
from expansion.scenarios import B, SCENARIOS, gap_pool, gap_topics, load, sub_instance  # noqa: E402


def value_of(A0, b0, AC, bC, d, selected):
    return fl.scratch_value(
        np.hstack([A0] + [AC[c][:, None].astype(np.int64) for c in selected]),
        d,
        np.concatenate([b0, [bC[c] for c in selected]]).astype(np.int64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", type=Path)
    ap.add_argument("tag")
    ap.add_argument("scenario", choices=SCENARIOS)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--limit", type=float, default=600.0, help="seconds per MIP")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    spec, A = load(a.instances, a.tag)
    pool_gap = gap_pool(spec, A, gap_topics(a.instances, a.tag))[1] if a.scenario == "topic_gap" else None
    rows = []
    t_all = time.time()
    for rep in range(a.reps):
        rng = np.random.default_rng(10_000 + rep)
        A0, b0, AC, bC, d, ks, n_dummy = sub_instance(a.scenario, rng, spec, A, pool_gap=pool_gap)
        k_max = max(ks)
        P0 = fl.Panel(A0, d, b0, extra=k_max, bmax=int(max(b0.max(), bC.max())))
        V0 = P0.solve()
        gr = greedy(copy.deepcopy(P0), AC, bC, k_max, n_dummy=n_dummy)
        retrieved = topk_retrieval(AC, bC, k_max, B)
        for k in ks:
            row = {
                "tag": a.tag,
                "scenario": a.scenario,
                "rep": rep,
                "budget": k,
                "submissions": int(len(d)),
                "candidates": int(AC.shape[0]),
                "certgreedy": value_of(A0, b0, AC, bC, d, gr.selected[:k]) - V0,
                "retrieval": value_of(A0, b0, AC, bC, d, retrieved[:k]) - V0,
                "stochastic": float(
                    np.mean(
                        [sum(stochastic_greedy(copy.deepcopy(P0), AC, bC, k, 0.1, np.random.default_rng(100 + s)).gains) for s in range(5)]
                    )
                ),
            }
            mip = mip_optimum(A0, b0, AC.astype(np.int64), bC, d, k, time_limit=a.limit, mip_gap=0.0, strong=True)
            sel = mip.get("selected", [])
            row["mip"] = value_of(A0, b0, AC, bC, d, sel) - V0 if sel else 0
            row["mip_bound"] = mip.get("bound", float("nan")) - V0
            row["mip_seconds"], row["closed"] = mip["seconds"], bool(mip["status"] == 0)
            assert row["certgreedy"] <= row["mip_bound"] + 1e-6 * abs(mip.get("bound", 0)) + 1, ("CertGreedy above the MIP bound", row)
            row["optimum"] = max(row["mip"], row["certgreedy"]) if row["closed"] else row["mip_bound"]
            rows.append(row)
            print(
                f"{a.tag} {a.scenario} rep {rep} k={k}: CertGreedy/OPT {row['certgreedy'] / max(1, row['optimum']):.4f}, "
                f"retrieval/OPT {row['retrieval'] / max(1, row['optimum']):.4f}, MIP {mip['seconds']:.1f}s, closed {row['closed']}",
                flush=True,
            )
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{a.tag}__{a.scenario}.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
