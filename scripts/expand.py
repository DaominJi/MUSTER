#!/usr/bin/env python3
"""Panel expansion on one instance: CertGreedy, its exact-marginal greedy variants and the baselines.

usage: python scripts/expand.py INSTANCE_DIR TAG {initialization,topic_gap,emergency} [--seed 0]
           [--methods retrieval,dual_only,lazy,incremental,naive,stochastic] [--stochastic-runs 3]
           [--max-seconds 86400] [--real-unpruned 900] [--out runs/expansion]

Methods, named as in the paper's panel-expansion experiments (CertGreedy always runs, the others are optional):
  retrieval    Top-k retrieval: candidates ranked by summed affinity
  dual_only    CertGreedy without lazy evaluation
  lazy         Lazy Greedy: lazy evaluation without dual bounds
  incremental  Incremental-Flow Greedy: InsertReviewer for every candidate in every round
  naive        Naive Greedy: a from-scratch assignment for every candidate in every round
  stochastic   Stochastic Greedy (epsilon 0.1), --stochastic-runs runs per budget

Gains are exact: the value of the reoptimized assignment of the enlarged panel minus the incumbent value, printed in
units of affinity. Calls count exact marginal evaluations. For timings, run with NUMBA_NUM_THREADS=1 on an idle machine.

`incremental` and `naive` evaluate every remaining candidate in every round, which is too slow on large instances. The
script therefore reports their cost as the exact number of evaluations times the time of one evaluation: InsertReviewer,
sampled at five stages of the greedy run and interpolated in between, or a from-scratch OR-Tools solve, the mean of the
incumbent panel and of a sample at the final panel. Each of the two also runs for real when this estimate is below
--real-unpruned seconds. A greedy variant stops after the round in which its clock passes --max-seconds.
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
from expansion.baselines import cand_scores, naive_greedy, ranking, stochastic_greedy  # noqa: E402
from expansion.certgreedy import greedy  # noqa: E402
from expansion.scenarios import B, SCENARIOS, build  # noqa: E402

METHODS = ("retrieval", "dual_only", "lazy", "incremental", "naive", "stochastic")
EPSILON = 0.1  # Stochastic Greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", type=Path, help="directory with the instance files")
    ap.add_argument("tag")
    ap.add_argument("scenario", choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=0, help="seed of the sampled panel")
    ap.add_argument("--methods", default="retrieval,dual_only,lazy,incremental,naive", help="comma-separated, from: " + ", ".join(METHODS))
    ap.add_argument("--stochastic-runs", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=24 * 3600.0, help="clock limit of one greedy variant")
    ap.add_argument(
        "--real-unpruned", type=float, default=900.0, help="run an unpruned variant when its estimate is below this many seconds"
    )
    ap.add_argument("--timing-sample", type=int, default=100, help="candidates timed per stage for the cost of one insertion")
    ap.add_argument("--scratch-sample", type=int, default=5, help="from-scratch solves timed at the final panel")
    ap.add_argument("--out", type=Path, default=None, help="directory for the JSON record of the run")
    a = ap.parse_args()
    methods = [m for m in a.methods.split(",") if m]
    assert all(m in METHODS for m in methods), f"methods must come from {METHODS}"
    name = f"{a.tag}__{a.scenario}__s{a.seed}"
    t0 = time.perf_counter()
    cfg = build(a.instances, a.tag, a.scenario, a.seed)
    A0, b0, AC, bC, d, n_dummy = (cfg[k] for k in ("A0", "b0", "AC", "bC", "d", "n_dummy"))
    ks = tuple(sorted(int(k) for k in cfg["ks"]))
    k_max = ks[-1]
    fl.scratch_value(np.zeros((1, 1), np.int64), np.ones(1, np.int64), np.ones(1, np.int64))  # loads OR-Tools before any timing
    P0 = fl.Panel(A0, d, b0, extra=k_max, bmax=int(max(b0.max(), bC.max())))
    t1 = time.perf_counter()
    V0 = P0.solve()
    t_ssp = time.perf_counter() - t1
    t1 = time.perf_counter()
    V0_ort = fl.scratch_value(A0, d, b0)
    t_ort = time.perf_counter() - t1
    assert V0 == V0_ort and P0.check() >= 0 and P0.dual_objective() == V0, "the incumbent assignment failed its checks"
    act = fl.usable(AC, bC)
    n_use = int(act.sum())
    print(
        f"[{name}] submissions {A0.shape[0]}, incumbents {A0.shape[1] - n_dummy}, candidates {n_use}, budgets {ks} "
        f"(built in {time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

    # untimed: compiles every function
    greedy(copy.deepcopy(P0), AC, bC, 1, ub_js=(1,), n_dummy=n_dummy)
    ranking(cand_scores(AC, B)[1], act)

    out = {
        "tag": a.tag,
        "scenario": a.scenario,
        "seed": a.seed,
        "budgets": list(ks),
        "candidates": n_use,
        "incumbent_value": int(V0),
        "reviews": int(d.sum()),
        "submissions": int(A0.shape[0]),
        "incumbents": int(A0.shape[1] - n_dummy),
        "scale": int(fl.SCALE),
        "instance": cfg["info"],
        "initial_solve_seconds": {"successive_shortest_paths": t_ssp, "ortools": t_ort},
    }
    rows = []  # (budget, method, gain, calls, seconds, note)

    def gains_of(order):
        """Exact gains of a fixed recruitment order at every budget."""
        store = {}
        fl.evaluate_sequence(
            copy.deepcopy(P0),
            AC,
            bC,
            [int(c) for c in order[:k_max]],
            k_max,
            "gains",
            checkpoints=set(ks),
            on_checkpoint=lambda j, p: store.__setitem__(int(j), int(p.value - V0)),
        )
        return store

    def table(run):
        """Gain, seconds and calls of a greedy run after every budget (None beyond the rounds of a run that was cut)."""
        res = {}
        for k in ks:
            j = min(k, len(run.selected))
            if run.timed_out and k > len(run.selected):
                res[k] = None
                continue
            whole = j == len(run.selected)  # the budget covers the whole run (also when no candidate could help any more)
            res[k] = {
                "gain": int(sum(run.gains[:j])),
                "rounds": j,
                "seconds": float(run.seconds if whole else run.cum_seconds[j - 1]),
                "calls": int(run.trials if whole else run.cum_trials[j - 1]),
            }
        return res

    def add_rows(method, at):
        for k in ks:
            r = at[k]
            rows.append((k, method, None, None, None, "cut") if r is None else (k, method, r["gain"], r.get("calls"), r.get("seconds"), ""))

    # ------------------------------------------------------------------------------------------------ CertGreedy
    g = greedy(copy.deepcopy(P0), AC, bC, k_max, ub_js=ks, n_dummy=n_dummy)
    n_sel = len(g.selected)
    out["certgreedy"] = {
        "at": table(g),
        "selected": g.selected,
        "rounds": n_sel,
        "stopped_early": g.stopped_early,
        "bound_seconds": g.bound_seconds,
        "optimum_upper_bound": {int(j): int(v) for j, v in g.ub.items()},
        "max_round_calls": int(max(g.per_round_trials)),
    }
    add_rows("CertGreedy", out["certgreedy"]["at"])
    print(f"  CertGreedy: {n_sel} recruits, {g.trials} calls, {g.seconds:.2f}s (bounds {g.bound_seconds:.2f}s)", flush=True)

    # ------------------------------------------------------------------------------------------- top-k retrieval
    if "retrieval" in methods:
        ta = time.perf_counter()
        order = ranking(cand_scores(AC, B)[1], act)
        t_sel = time.perf_counter() - ta
        gs = gains_of(order)
        out["retrieval"] = {"seconds": t_sel, "at": {k: {"gain": gs[k], "seconds": t_sel} for k in ks}}
        add_rows("Top-k retrieval", out["retrieval"]["at"])

    # ------------------------------------------------------------------------------ CertGreedy without lazy evaluation
    if "dual_only" in methods:
        ge = greedy(copy.deepcopy(P0), AC, bC, k_max, n_dummy=n_dummy, lazy=False, max_seconds=a.max_seconds)
        assert ge.selected == g.selected[: len(ge.selected)], "the variant without lazy evaluation selected different recruits"
        out["dual_only"] = {
            "at": table(ge),
            "timed_out": ge.timed_out,
            "rounds": len(ge.selected),
            "seconds": ge.seconds,
            "calls": int(ge.trials),
        }
        add_rows("CertGreedy w/o lazy evaluation", out["dual_only"]["at"])
        print(f"  w/o lazy evaluation: {ge.trials} calls, {ge.seconds:.2f}s{' (cut)' if ge.timed_out else ''}", flush=True)

    # --------------------------------------------------------------------------------------------------- Lazy Greedy
    if "lazy" in methods:
        gd = greedy(copy.deepcopy(P0), AC, bC, k_max, n_dummy=n_dummy, dual=False, max_seconds=a.max_seconds)
        assert gd.selected == g.selected[: len(gd.selected)], "Lazy Greedy selected different recruits"
        out["lazy"] = {
            "at": table(gd),
            "timed_out": gd.timed_out,
            "rounds": len(gd.selected),
            "seconds": gd.seconds,
            "calls": int(gd.trials),
            "first_round_calls": int(gd.per_round_trials[0]),
        }
        add_rows("Lazy Greedy", out["lazy"]["at"])
        print(f"  Lazy Greedy: {gd.trials:,} calls, {gd.seconds:.1f}s{' (cut)' if gd.timed_out else ''}", flush=True)

    # --------------------------------------------------------------------- unpruned variants: cost of every evaluation
    if "incremental" in methods or "naive" in methods:
        stages = sorted({int(round(f * n_sel)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)})
        sample = np.random.default_rng(7).choice(np.flatnonzero(act), size=min(a.timing_sample, n_use), replace=False)
        ins = {}

        def probe(j, panel):
            ta = time.perf_counter()
            for c in sample:
                panel.trial(AC[c], bC[c])
            ins[int(j)] = (time.perf_counter() - ta) / len(sample)

        Pg = copy.deepcopy(P0)
        fl.evaluate_sequence(Pg, AC, bC, g.selected, n_sel, "probe", checkpoints=set(stages), on_checkpoint=probe)
        xs = sorted(ins)
        per_round = np.interp(np.arange(max(n_sel, 1)), xs, [ins[x] for x in xs])  # seconds per insertion in round i
        calls = np.array([n_use - i for i in range(max(n_sel, 1))], np.float64)  # evaluations of unpruned greedy in round i
        cum_calls, cum_ins = np.cumsum(calls), np.cumsum(calls * per_round)
        ts = []
        for c in sample[: a.scratch_sample]:
            ta = time.perf_counter()
            fl.scratch_value(np.hstack([Pg.A[:, : Pg.nW], AC[c][:, None].astype(np.int64)]), d, np.append(Pg.cap[: Pg.nW], B))
            ts.append(time.perf_counter() - ta)
        per_solve = 0.5 * (float(np.mean(ts)) + t_ort) if ts else t_ort
        est = {
            k: {
                "calls": int(cum_calls[max(1, min(k, n_sel)) - 1]),
                "insertion_seconds": float(cum_ins[max(1, min(k, n_sel)) - 1]),
                "resolve_seconds": float(cum_calls[max(1, min(k, n_sel)) - 1] * per_solve),
            }
            for k in ks
        }
        out["unpruned_estimate"] = {"at": est, "insertion_seconds_per_call": ins, "resolve_seconds_per_call": per_solve}
        gain_at = {k: out["certgreedy"]["at"][k]["gain"] for k in ks}  # an unpruned variant returns CertGreedy's recruits
        if "incremental" in methods:
            if est[k_max]["insertion_seconds"] <= a.real_unpruned:
                gf = greedy(copy.deepcopy(P0), AC, bC, k_max, n_dummy=n_dummy, lazy=False, dual=False, max_seconds=a.max_seconds)
                assert gf.selected == g.selected[: len(gf.selected)], "Incremental-Flow Greedy selected different recruits"
                out["incremental"] = {"at": table(gf), "timed_out": gf.timed_out, "seconds": gf.seconds, "calls": int(gf.trials)}
                add_rows("Incremental-Flow Greedy", out["incremental"]["at"])
            else:
                for k in ks:
                    rows.append((k, "Incremental-Flow Greedy", gain_at[k], est[k]["calls"], est[k]["insertion_seconds"], "estimated"))
        if "naive" in methods:
            if est[k_max]["resolve_seconds"] <= a.real_unpruned:
                gn, _ = naive_greedy(A0, b0, AC, bC, d, k_max)
                assert gn.selected == g.selected[: len(gn.selected)], "Naive Greedy selected different recruits"
                out["naive"] = {"seconds": gn.seconds, "calls": int(gn.trials), "gain": int(sum(gn.gains)), "rounds": len(gn.selected)}
                rows.append((k_max, "Naive Greedy", int(sum(gn.gains)), int(gn.trials), gn.seconds, ""))
            else:
                for k in ks:
                    rows.append((k, "Naive Greedy", gain_at[k], est[k]["calls"], est[k]["resolve_seconds"], "estimated"))

    # ---------------------------------------------------------------------------------------------- Stochastic Greedy
    if "stochastic" in methods:
        stochastic_greedy(copy.deepcopy(P0), AC[:64], bC[:64], 1, EPSILON, np.random.default_rng(0))  # untimed: compiles
        out["stochastic"] = {}
        for k in ks:
            runs = []
            for r in range(a.stochastic_runs):
                sr = stochastic_greedy(copy.deepcopy(P0), AC, bC, k, EPSILON, np.random.default_rng(2000 + 17 * r + a.seed))
                runs.append({"gain": int(sum(sr.gains)), "calls": int(sr.trials), "seconds": sr.seconds})
            out["stochastic"][k] = runs
            rows.append(
                (
                    k,
                    f"Stochastic Greedy (mean of {len(runs)})",
                    float(np.mean([x["gain"] for x in runs])),
                    float(np.mean([x["calls"] for x in runs])),
                    float(np.mean([x["seconds"] for x in runs])),
                    "",
                )
            )

    print(f"\n{'budget':>6}  {'method':32s} {'gain':>10} {'calls':>12} {'seconds':>10}")
    for k, method, gain, n_calls, sec, note in sorted(rows, key=lambda r: r[0]):
        f_gain = "" if gain is None else f"{gain / fl.SCALE:.4f}"
        f_calls = "" if n_calls is None else f"{n_calls:,.0f}"
        f_sec = "" if sec is None else f"{sec:.2f}"
        print(f"{k:>6}  {method:32s} {f_gain:>10} {f_calls:>12} {f_sec:>10}  {note}")
    out["seconds_total"] = time.perf_counter() - t0
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {a.out / (name + '.json')}")


if __name__ == "__main__":
    main()
