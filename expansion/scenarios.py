"""The three recruitment scenarios of the experiments, built from one venue instance.

Instance files (docs/EXPANSION.md):
  <tag>_A.npy       (reviewers, submissions) int32 affinities in micro-units, -1 = ineligible (conflict of interest)
  <tag>_spec.npz    inc_pool: one bool per reviewer, True for the venue's earlier authors (the incumbent pool)
  topic_<tag>.npz   sub, past: unit vectors of the submissions and of the venue's earlier papers (topic gap only)

Every submission needs D reviews and every reviewer takes at most B submissions.
  initialization  no incumbent panel (zero-affinity placeholder reviewers keep the assignment feasible), every reviewer
                  is a candidate, budget ceil((1 + SIGMA) T / B)
  topic_gap       the incumbent panel is sampled from the incumbent pool after leaving out the GAP_REMOVE share best
                  suited to the N_GAP fastest-growing topics, and holds spare capacity SIGMA
  emergency       a WITHDRAW share of the members of a full panel withdraw after the optimal assignment. The instance
                  holds the lost reviews, the remaining members' spare capacity and placeholders
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

from . import flow as fl

D, B, SIGMA = 3, 4, 0.10  # reviews per submission, submissions per reviewer, spare capacity of a panel
N_TOPICS, N_GAP = 10, 2  # topics of a venue, and how many of the fastest-growing ones form the gap
GAP_REMOVE = 0.5  # share of the incumbent pool left out of a topic-gap panel: the members best suited to the gap topics
WITHDRAW = 0.05  # share of the assigned members who withdraw in an emergency
N_DUMMY = 3  # zero-affinity placeholder reviewers, one per required review of a submission
SCENARIOS = ("initialization", "topic_gap", "emergency")
BUDGETS = {"topic_gap": (10, 20, 50, 100, 200), "emergency": (5, 10, 20, 50)}
SUB_Q, SUB_C = 100, 500  # submissions and candidates of a sub-instance
SUB_BUDGETS = (2, 5, 10, 20)


def load(inst_dir, tag):
    """The specification arrays and the memory-mapped affinity matrix (reviewers x submissions) of a venue."""
    inst_dir = Path(inst_dir)
    spec = dict(np.load(inst_dir / f"{tag}_spec.npz", allow_pickle=False))
    return spec, np.load(inst_dir / f"{tag}_A.npy", mmap_mode="r")


@njit(cache=True, parallel=True)
def topb_on(AC, b, cols):
    """Sum of the b largest affinities of each candidate over the submissions cols only."""
    nC = AC.shape[0]
    out = np.zeros(nC, np.int64)
    for c in prange(nC):
        top = np.zeros(b, np.int64)
        for i in range(len(cols)):
            a = np.int64(AC[c, cols[i]])
            if a > top[b - 1]:
                j = b - 1
                while j > 0 and top[j - 1] < a:
                    top[j] = top[j - 1]
                    j -= 1
                top[j] = a
        out[c] = top.sum()
    return out


# ------------------------------------------------------------------------------------------------- topics (topic gap)
def spherical_kmeans(X, k, restarts=10, seed=0, iters=100):
    """Cosine k-means on unit vectors: k-means++ seeding, the best of `restarts` runs by the sum of cosines."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float64)
    n = len(X)
    best = None
    for _ in range(restarts):
        cent = [X[int(rng.integers(n))]]
        for _j in range(1, k):
            d2 = np.maximum(1.0 - np.max(X @ np.asarray(cent).T, axis=1), 0.0)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            cent.append(X[int(rng.choice(n, p=p))])
        C = np.asarray(cent)
        lab = None
        for _it in range(iters):
            new = np.argmax(X @ C.T, axis=1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                mj = lab == j
                if mj.any():
                    v = X[mj].sum(0)
                    C[j] = v / max(np.linalg.norm(v), 1e-12)
        obj = float(np.max(X @ C.T, axis=1).sum())
        if best is None or obj > best[0]:
            best = (obj, C.copy(), np.argmax(X @ C.T, axis=1))
    return best[1], best[2], best[0]


def gap_topics(inst_dir, tag):
    """Topics of the submissions, and the N_GAP topics whose share grew most against the venue's earlier papers.

    When gap_topics_<tag>.npz is present, its stored clustering is returned, so that every machine builds the same
    instances (the floating-point sums of k-means can differ in their last digits across platforms)."""
    inst_dir = Path(inst_dir)
    cache = inst_dir / f"gap_topics_{tag}.npz"
    if cache.exists():
        with np.load(cache) as zc:
            return {key: (zc[key] if zc[key].ndim else zc[key].item()) for key in zc.files}
    z = np.load(inst_dir / f"topic_{tag}.npz")
    C, lab, obj = spherical_kmeans(z["sub"], N_TOPICS)
    past_lab = np.argmax(np.asarray(z["past"], np.float64) @ C.T, axis=1)
    n_sub = np.bincount(lab, minlength=N_TOPICS)
    n_past = np.bincount(past_lab, minlength=N_TOPICS)
    growth = ((n_sub + 1) / (len(lab) + N_TOPICS)) / ((n_past + 1) / (len(past_lab) + N_TOPICS))
    gap = np.argsort(-growth, kind="stable")[:N_GAP]
    return {
        "labels": lab,
        "past_labels": past_lab,
        "n_sub": n_sub,
        "n_past": n_past,
        "growth": growth,
        "gap": np.sort(gap),
        "objective": obj,
    }


def gap_pool(spec, A, tp):
    """The gap-topic submissions, and the incumbent pool of a panel that covers them thinly: the pool is ranked by the sum
    of each member's B largest affinities over the gap-topic submissions, and the best-suited GAP_REMOVE share is left out."""
    gap_sub = np.flatnonzero(np.isin(tp["labels"], tp["gap"])).astype(np.int64)
    pool = np.flatnonzero(spec["inc_pool"])
    e_gap = topb_on(np.ascontiguousarray(A[pool]), B, gap_sub)
    order = np.lexsort((pool, -e_gap))  # best suited first, ties by index
    return gap_sub, np.sort(pool[order[int(round(GAP_REMOVE * len(pool))) :]])


# ------------------------------------------------------------------------------------------------------------- panels
def sample_panel(pool, T, seed, b=B, sigma=SIGMA):
    """A random sample of the pool whose total capacity first reaches (1 + sigma) T."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool)
    n0 = int(math.ceil((1 + sigma) * T / b - 1e-9))
    if n0 > len(perm):
        raise ValueError(f"pool too small: need {n0}, have {len(perm)}")
    return np.sort(perm[:n0])


def dummy_columns(nQ):
    """Affinities and capacities of the placeholder reviewers: zero affinity, eligible for everything, unlimited capacity."""
    return np.zeros((nQ, N_DUMMY), np.int64), np.full(N_DUMMY, nQ, np.int64)


def _withdraw(Af, n_members, rng):
    """Solve the full panel Af (submissions x members), let a WITHDRAW share of its assigned members withdraw, and return
    the emergency instance: (A0, b0, Qp, dq, info). A0 holds the placeholders and then the remaining members with spare
    capacity, over the submissions Qp that lost dq reviews."""
    nQ = Af.shape[0]
    Pf = fl.Panel(Af, np.full(nQ, D, np.int64), np.full(n_members, B, np.int64), extra=0, bmax=B)
    t0 = time.perf_counter()
    Vf = Pf.solve()
    t_full = time.perf_counter() - t0
    assert Pf.check() >= 0 and Pf.dual_objective() == Vf
    asg, load_ = Pf.base[2], Pf.base[3]
    holders = np.flatnonzero(load_[:n_members] > 0)
    nw = max(1, int(round(WITHDRAW * len(holders))))
    gone = np.sort(rng.choice(holders, size=nw, replace=False))
    lost_q = np.concatenate([asg[w, : load_[w]] for w in gone]).astype(np.int64)
    lost_w = np.concatenate([np.full(load_[w], w) for w in gone]).astype(np.int64)
    lost_aff = int(Af[lost_q, lost_w].sum())
    Qp, dq = np.unique(lost_q, return_counts=True)
    stay = np.setdiff1d(np.arange(n_members), gone)
    stay = stay[load_[stay] < B]  # members with spare capacity
    Ar = Af[np.ix_(Qp, stay)].copy()
    qpos = {int(q): i for i, q in enumerate(Qp)}
    for j, w in enumerate(stay):  # a member already reviewing a submission cannot take a second slot of it
        for q in asg[w, : load_[w]]:
            if int(q) in qpos:
                Ar[qpos[int(q)], j] = -1
    Z, bz = dummy_columns(len(Qp))
    A0 = np.ascontiguousarray(np.hstack([Z, Ar]))
    b0 = np.concatenate([bz, (B - load_[stay]).astype(np.int64)])
    info = {
        "full_panel": int(n_members),
        "full_value_mean_affinity": Vf / (D * nQ) / fl.SCALE,
        "full_solve_seconds": t_full,
        "withdrawn": int(nw),
        "lost_reviews": int(len(lost_q)),
        "lost_mean_affinity": lost_aff / len(lost_q) / fl.SCALE,
        "urgent_submissions": int(len(Qp)),
        "members_with_spare": int(len(stay)),
        "spare_slots": int((B - load_[stay]).sum()),
        "lost_aff": lost_aff,
    }
    return A0, b0, Qp, dq.astype(np.int64), info


def build(inst_dir, tag, scenario, seed=0):
    """One instance: a dict with the incumbent affinities A0 (submissions x incumbents, placeholders first), their
    capacities b0, the candidates' affinities AC (candidates x submissions) and capacities bC, the demands d, the budgets
    ks, the number of placeholder reviewers n_dummy, the candidates' reviewer indices C and instance statistics info."""
    spec, A = load(inst_dir, tag)
    nU, nQ = A.shape
    T = D * nQ
    info = {"universe": int(nU), "submissions": int(nQ)}
    if scenario == "initialization":
        A0, b0 = dummy_columns(nQ)
        C = np.arange(nU)
        AC = np.ascontiguousarray(A)
        d = np.full(nQ, D, np.int64)
        ks = (int(math.ceil((1 + SIGMA) * T / B - 1e-9)),)
        n_dummy = N_DUMMY
    elif scenario == "topic_gap":
        tp = gap_topics(inst_dir, tag)
        gap_sub, pool = gap_pool(spec, A, tp)
        R0 = sample_panel(pool, T, seed)
        C = np.setdiff1d(np.arange(nU), R0)
        A0 = np.ascontiguousarray(np.asarray(A[R0]).T).astype(np.int64)
        b0 = np.full(len(R0), B, np.int64)
        AC = np.ascontiguousarray(A[C])
        d = np.full(nQ, D, np.int64)
        ks = BUDGETS["topic_gap"]
        n_dummy = 0
        info.update(
            {
                "topics": {"n_sub": tp["n_sub"].tolist(), "n_past": tp["n_past"].tolist(), "gap": tp["gap"].tolist()},
                "gap_submissions": int(len(gap_sub)),
                "incumbent_pool": int(spec["inc_pool"].sum()),
                "pool_after_removal": int(len(pool)),
            }
        )
    elif scenario == "emergency":
        pool = np.flatnonzero(spec["inc_pool"])
        Rf = sample_panel(pool, T, seed)
        Af = np.ascontiguousarray(np.asarray(A[Rf]).T).astype(np.int64)
        A0, b0, Qp, d, winfo = _withdraw(Af, len(Rf), np.random.default_rng(100 + seed))
        C = np.setdiff1d(np.arange(nU), Rf)
        AC = np.ascontiguousarray(np.asarray(A[:, Qp])[C])
        ks = BUDGETS["emergency"]
        n_dummy = N_DUMMY
        info.update(winfo)
    else:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    return dict(A0=A0, b0=b0, AC=AC, bC=np.full(AC.shape[0], B, np.int64), d=d, ks=tuple(ks), n_dummy=n_dummy, C=C, info=info)


def sub_instance(scenario, rng, spec, A, tp=None, pool_gap=None):
    """A sub-instance with SUB_Q random submissions (all of them if the venue has fewer) and SUB_C random candidates, small
    enough for the MIP over all candidates. Returns (A0, b0, AC, bC, d, budgets, n_dummy).

    initialization  no incumbent (placeholders), budget ceil((1 + SIGMA) T / B)
    topic_gap       incumbents sampled from pool_gap (the pool of `gap_pool`), budgets SUB_BUDGETS
    emergency       a full panel on the sub-instance, of which a WITHDRAW share withdraws (at least one member), budgets
                    SUB_BUDGETS up to the number of lost reviews"""
    nU, nQ = A.shape
    qs = np.sort(rng.choice(nQ, size=min(SUB_Q, nQ), replace=False))
    T = D * len(qs)
    n0 = int(math.ceil((1 + SIGMA) * T / B - 1e-9))
    bC = np.full(SUB_C, B, np.int64)
    if scenario == "initialization":
        A0, b0 = dummy_columns(len(qs))
        C = np.sort(rng.choice(nU, size=SUB_C, replace=False))
        d = np.full(len(qs), D, np.int64)
        AC = np.ascontiguousarray(np.asarray(A[C])[:, qs])
        return A0, b0, AC, bC, d, (n0,), N_DUMMY
    pool = pool_gap if scenario == "topic_gap" else np.flatnonzero(spec["inc_pool"])
    for _ in range(50):
        R0 = np.sort(rng.permutation(pool)[:n0])
        C = np.sort(rng.choice(np.setdiff1d(np.arange(nU), R0), size=SUB_C, replace=False))
        A0 = np.ascontiguousarray(np.asarray(A[R0])[:, qs].T).astype(np.int64)
        AC = np.ascontiguousarray(np.asarray(A[C])[:, qs])
        d = np.full(len(qs), D, np.int64)
        b0 = np.full(n0, B, np.int64)
        if fl.scratch_value(A0, d, b0) is None:
            continue
        if scenario == "topic_gap":
            return A0, b0, AC, bC, d, SUB_BUDGETS, 0
        A0e, b0e, Qp, dq, winfo = _withdraw(A0, n0, rng)
        ks = tuple(k for k in SUB_BUDGETS if k <= winfo["lost_reviews"])
        return A0e, b0e, np.ascontiguousarray(AC[:, Qp]), bC, dq, ks, N_DUMMY
    raise RuntimeError("no feasible sub-instance")
