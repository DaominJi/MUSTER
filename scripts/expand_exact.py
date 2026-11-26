#!/usr/bin/env python3
"""Exact optimum of panel expansion on one instance, per budget, with HiGHS (through scipy).

usage: python scripts/expand_exact.py INSTANCE_DIR TAG {initialization,topic_gap,emergency} [--seed 0]
           [--budgets 5,10|all] [--limit 3600] [--full-mip] [--plain] [--out runs/expansion_exact]

By default the optimum is found by column generation over the candidates and the assignment edges (expansion/exact.py,
`sift`), which handles full venue instances. --full-mip solves the mixed-integer program over all candidates directly,
which suits small instances. Both start from the recruits of CertGreedy, and the value of every selection is recomputed
exactly with a from-scratch assignment. A budget is closed when its optimum is proven within --limit seconds. Otherwise
the best selection and the upper bound are reported. --plain leaves the cutoff row out of the deciding MIP (a cross-check).
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
from expansion.certgreedy import greedy  # noqa: E402
from expansion.exact import Problem, mip_optimum, sift  # noqa: E402
from expansion.scenarios import SCENARIOS, build  # noqa: E402


def value_of(A0, b0, AC, bC, d, selected):
    """Exact assignment value of the incumbent panel plus the selected candidates."""
    return fl.scratch_value(
        np.hstack([A0] + [AC[c][:, None].astype(np.int64) for c in selected]),
        d,
        np.concatenate([b0, [bC[c] for c in selected]]).astype(np.int64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", type=Path, help="directory with the instance files")
    ap.add_argument("tag")
    ap.add_argument("scenario", choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="all", help="comma-separated budgets, or all")
    ap.add_argument("--limit", type=float, default=3600.0, help="seconds per budget")
    ap.add_argument("--full-mip", action="store_true", help="the MIP over all candidates instead of column generation")
    ap.add_argument("--plain", action="store_true", help="no cutoff row in the deciding MIP (cross-check)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    name = f"{a.tag}__{a.scenario}__s{a.seed}"
    t0 = time.perf_counter()
    cfg = build(a.instances, a.tag, a.scenario, a.seed)
    A0, b0, AC, bC, d, ks, n_dummy = (cfg[k] for k in ("A0", "b0", "AC", "bC", "d", "ks", "n_dummy"))
    budgets = sorted(int(k) for k in ks) if a.budgets == "all" else [int(x) for x in a.budgets.split(",")]
    k_max = max(max(int(k) for k in ks), max(budgets))
    P0 = fl.Panel(A0, d, b0, extra=k_max, bmax=int(max(b0.max(), bC.max())))
    V0 = P0.solve()
    bC = np.asarray(bC, np.int64)
    g = greedy(copy.deepcopy(P0), AC, bC, k_max, ub_js=tuple(budgets), n_dummy=n_dummy)
    # the edges of the incumbent optimum between real incumbents and submissions (placeholders have unlimited capacity)
    free = np.asarray(b0) >= A0.shape[0]
    inc_pos = -np.ones(A0.shape[1], np.int64)
    inc_pos[np.flatnonzero(~free)] = np.arange(int((~free).sum()))
    pairs = [(inc_pos[m], q) for q, m in P0.assignment() if not isinstance(m, tuple) and inc_pos[m] >= 0]
    start = (np.array([p[0] for p in pairs], np.int64), np.array([p[1] for p in pairs], np.int64))
    print(
        f"[{name}] candidates {AC.shape[0]}, incumbents {int((~free).sum())}, budgets {budgets}, limit {a.limit:.0f}s "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    records = []
    for k in budgets:
        g_gain = int(sum(g.gains[:k]))
        if a.full_mip:
            m = mip_optimum(A0, b0, AC.astype(np.int64), bC, d, k, time_limit=a.limit, mip_gap=0.0, strong=True)
            sel = m.get("selected", [])
            gain = max(int(value_of(A0, b0, AC, bC, d, sel)) - V0 if sel else 0, g_gain)
            rec = {
                "budget": k,
                "closed": bool(m["status"] == 0),
                "gain": gain,
                "bound_gain": float(m.get("bound", float("nan")) - V0),
                "seconds": m["seconds"],
                "method": "MIP over all candidates",
            }
        else:
            r = sift(Problem(A0, b0, AC, bC, d, k), V0, g.selected[:k], start, a.limit, plain=a.plain)
            rec = {
                "budget": k,
                "closed": bool(r["closed"]),
                "gain": int(r["value"] - V0),
                "bound_gain": float(r["bound"] - V0) if r["bound"] is not None else None,
                "seconds": r["seconds"],
                "rounds": r["rounds"],
                "method": "column generation over candidates and edges",
            }
        rec.update({"certgreedy_gain": g_gain, "certgreedy_upper_bound": int(g.ub[k]) if k in g.ub else None})
        if rec["closed"]:
            assert rec["gain"] >= g_gain, "the optimum is below the gain of CertGreedy"
            assert (
                rec["certgreedy_upper_bound"] is None or rec["gain"] <= rec["certgreedy_upper_bound"]
            ), "the optimum is above the a posteriori bound"
        records.append(rec)
        print(
            f"--- k={k}: {'optimum' if rec['closed'] else 'best found (open)'} {rec['gain'] / fl.SCALE:.4f}, CertGreedy {g_gain / fl.SCALE:.4f} "
            f"({100 * g_gain / max(1, rec['gain']):.2f}%), {rec['seconds']:.0f}s",
            flush=True,
        )
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{name}.json").write_text(
            json.dumps(
                {
                    "tag": a.tag,
                    "scenario": a.scenario,
                    "seed": a.seed,
                    "incumbent_value": int(V0),
                    "scale": int(fl.SCALE),
                    "budgets": records,
                },
                indent=1,
                default=float,
            )
        )


if __name__ == "__main__":
    main()
