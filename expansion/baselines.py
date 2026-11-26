"""Baselines of the panel-expansion experiments: top-k retrieval, Stochastic Greedy and Naive Greedy."""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from .flow import Run, scratch_value, usable


@njit(cache=True, parallel=True)
def cand_scores(AC, b):
    """Per candidate (row of AC): maximum, sum and sum of the b largest affinities over its eligible submissions."""
    nC, nQ = AC.shape
    smax = np.full(nC, -1, np.int64)
    ssum = np.zeros(nC, np.int64)
    stop = np.zeros(nC, np.int64)
    for c in prange(nC):
        top = np.zeros(b, np.int64)
        m = np.int64(-1)
        s = np.int64(0)
        for q in range(nQ):
            a = np.int64(AC[c, q])
            if a < 0:
                continue
            if a > m:
                m = a
            s += a
            if a > top[b - 1]:
                j = b - 1
                while j > 0 and top[j - 1] < a:
                    top[j] = top[j - 1]
                    j -= 1
                top[j] = a
        smax[c] = m
        ssum[c] = s
        stop[c] = top.sum()
    return smax, ssum, stop


def ranking(score, active):
    """Active candidates by decreasing score, ties by index."""
    s = np.where(active, score, np.iinfo(np.int64).min).astype(np.int64)
    order = np.lexsort((np.arange(len(s)), -s))
    return [int(c) for c in order if active[c]]


def topk_retrieval(AC, bC, k, b):
    """Top-k retrieval: the k usable candidates with the largest summed affinity over their eligible submissions."""
    return ranking(cand_scores(AC, b)[1], usable(AC, bC))[:k]


def stochastic_greedy(panel, AC, bC, k, eps, rng):
    """Stochastic greedy (Mirzasoleiman et al. 2015) with the exact insertion oracle."""
    active = usable(AC, bC)
    n = int(active.sum())
    s = int(math.ceil(n / k * math.log(1.0 / eps)))
    run = Run(method=f"stochastic-{eps}")
    d0, s0 = panel.stats.copy(), panel.n_trials
    t0 = time.perf_counter()
    for i in range(k):
        pool = np.flatnonzero(active)
        if len(pool) == 0:
            break
        sample = rng.choice(pool, size=min(len(pool), s), replace=False)
        sample.sort()
        best, best_c = -1, -1
        for c in sample:
            g = panel.trial(AC[c], bC[c])
            if g > best:
                best, best_c = g, int(c)
                panel.keep_trial_as_best(AC[c], bC[c])
        panel.commit_best(("c", best_c))
        active[best_c] = False
        run.selected.append(best_c)
        run.gains.append(int(best))
        run.per_round_trials.append(len(sample))
    run.seconds = time.perf_counter() - t0
    run.trials = panel.n_trials - s0
    run.dijkstra, run.settled = (panel.stats - d0).tolist()
    return run


def naive_greedy(A0, b0, AC, bC, d, k, max_rounds=None, sample=None, rng=None):
    """Naive Greedy: every marginal is a from-scratch OR-Tools solve of the trial panel.

    With `sample`, every round evaluates only `sample` random candidates (for timing). max_rounds stops early.
    Returns (Run, mean seconds per solve)."""
    active = usable(AC, bC)
    cols, caps = [A0], [np.asarray(b0)]
    base = scratch_value(A0, d, b0)
    run = Run(method="naive-greedy")
    times = []
    G = 0
    t0 = time.perf_counter()
    for i in range(k if max_rounds is None else min(k, max_rounds)):
        Acur, bcur = np.hstack(cols), np.concatenate(caps)
        cand = np.flatnonzero(active)
        if sample is not None:
            cand = np.sort(rng.choice(cand, size=min(sample, len(cand)), replace=False))
        best, best_c = 0, -1
        for c in cand:
            t1 = time.perf_counter()
            v = scratch_value(np.hstack([Acur, AC[c][:, None]]), d, np.append(bcur, min(bC[c], int((AC[c] >= 0).sum()))))
            times.append(time.perf_counter() - t1)
            g = v - base
            run.trials += 1
            if g > best:
                best, best_c = g, int(c)
        if best <= 0:
            run.stopped_early = True
            break
        cols.append(AC[best_c][:, None])
        caps.append(np.array([bC[best_c]]))
        active[best_c] = False
        base += best
        G += best
        run.selected.append(best_c)
        run.gains.append(int(best))
    run.seconds = time.perf_counter() - t0
    return run, float(np.mean(times)) if times else float("nan")
