"""CertGreedy: exact greedy recruitment with certified pruning (the paper's certified panel expansion), and its variants.

Every round selects the candidate with the largest exact marginal gain, computed by InsertReviewer. Two certificates
remove most evaluations without changing the selection:
  dual bounds   U_r = TopSum_{b_r} [a_qr - pi_q]_+ from the submission prices pi of an optimal dual of the current panel
  lazy bounds   exact marginals and bounds of earlier rounds, which stay valid because the gain is submodular
A candidate is evaluated only while its smaller bound can still beat the best exact marginal of the round. Ties go to
the smallest candidate index, so every variant returns the same recruits.

Submission prices. `Panel.pi` reads them from the potentials that the insertions leave. `tight_pi` computes the largest
prices of any optimal dual with one shortest-path pass from the sink. `priced_pi` extends them to panels that still use
placeholder reviewers (technical report, "Tight Shadow Prices and Priced Placeholders").
"""

from __future__ import annotations

import time

import numpy as np
from numba import get_num_threads, njit, prange

from . import flow as fl


# ---------------------------------------------------------------------------------------------- submission prices
@njit(cache=True)
def _dist_from_sink(nQ, nW, A, qasg, qcnt, asg, load, phiQ, phiW, phiT, dist, pred, hn, hpos, settled, mark):
    """Shortest reduced-cost distances from the sink t to every node of the residual network of the current optimal flow
    (the Dijkstra of InsertReviewer, without a target). Unreachable nodes keep INF."""
    nN = nQ + nW + 1
    for v in range(nN):
        dist[v] = fl.INF
        pred[v] = -2
        hpos[v] = -1
        settled[v] = 0
    size = 0
    for w in range(nW):
        if load[w] > 0:
            size = fl._relax(nQ + w, phiT[0] - phiW[w], dist, pred, -1, hn, hpos, size)
    while size > 0:
        u, size = fl._hpop(hn, hpos, dist, size)
        settled[u] = 1
        du = dist[u]
        if u < nQ:
            q = u
            pq = phiQ[q]
            for j in range(qcnt[q]):
                mark[qasg[q, j]] = 1
            for w in range(nW):
                aw = A[q, w]
                if aw >= 0 and mark[w] == 0:
                    v = nQ + w
                    if settled[v] == 0:
                        size = fl._relax(v, du - aw + pq - phiW[w], dist, pred, u, hn, hpos, size)
            for j in range(qcnt[q]):
                mark[qasg[q, j]] = 0
        else:
            w = u - nQ
            pw = phiW[w]
            for j in range(load[w]):
                q = asg[w, j]
                if settled[q] == 0:
                    size = fl._relax(q, du + A[q, w] + pw - phiQ[q], dist, pred, u, hn, hpos, size)


def tight_pi(panel):
    """The largest submission prices of any optimal dual of the current panel: pi_q = the cost of the cheapest
    rearrangement that frees one slot of q, i.e. the shortest distance from the sink t to q in the residual network.
    Every feasible potential phi has phi(q) - phi(t) <= dist(t -> q), and phi = dist(t -> .) is feasible, so these prices
    dominate the prices read from the potentials that the insertions leave (Panel.pi), and the bound
    U_r = TopSum_{b_r} [a_qr - pi_q]_+ is the smallest of its family. A submission that t cannot reach
    has no slot that can be freed and gets a price above every affinity."""
    qasg, qcnt, asg, load, phiQ, phiW, phiT = panel.base
    _dist_from_sink(
        panel.nQ,
        panel.nW,
        panel.A,
        qasg,
        qcnt,
        asg,
        load,
        phiQ,
        phiW,
        phiT,
        panel.dist,
        panel.pred,
        panel.hn,
        panel.hpos,
        panel.settled,
        panel.mark,
    )
    dq = panel.dist[: panel.nQ]
    return np.where(dq < fl.INF, dq + phiQ - phiT[0], np.int64(4 * fl.SCALE))


@njit(cache=True)
def _dist_from_sink_priced(nQ, nW, A, qasg, qcnt, asg, load, phiQ, phiW, phiT, dist, pred, hn, hpos, settled, mark, n_dummy, rho):
    """_dist_from_sink on the residual network of the assignment LP in which the placeholder reviews of the current
    assignment are worth rho and no further review may go to a placeholder: releasing a placeholder's review costs rho,
    and the arcs from a submission to a placeholder are left out. The current potentials stay feasible there (arcs are
    only removed and costs only grow), so Dijkstra on their reduced costs is valid."""
    nN = nQ + nW + 1
    for v in range(nN):
        dist[v] = fl.INF
        pred[v] = -2
        hpos[v] = -1
        settled[v] = 0
    size = 0
    for w in range(nW):
        if load[w] > 0:
            size = fl._relax(nQ + w, phiT[0] - phiW[w], dist, pred, -1, hn, hpos, size)
    while size > 0:
        u, size = fl._hpop(hn, hpos, dist, size)
        settled[u] = 1
        du = dist[u]
        if u < nQ:
            q = u
            pq = phiQ[q]
            for j in range(qcnt[q]):
                mark[qasg[q, j]] = 1
            for w in range(n_dummy, nW):
                aw = A[q, w]
                if aw >= 0 and mark[w] == 0:
                    v = nQ + w
                    if settled[v] == 0:
                        size = fl._relax(v, du - aw + pq - phiW[w], dist, pred, u, hn, hpos, size)
            for j in range(qcnt[q]):
                mark[qasg[q, j]] = 0
        else:
            w = u - nQ
            pw = phiW[w]
            extra = rho if w < n_dummy else 0
            for j in range(load[w]):
                q = asg[w, j]
                if settled[q] == 0:
                    size = fl._relax(q, du + A[q, w] + pw - phiQ[q] + extra, dist, pred, u, hn, hpos, size)


def priced_pi(panel, n_dummy, rho):
    """Submission prices of an optimal dual of the assignment LP in which the m placeholder reviews of the current
    assignment are worth rho each and no further review may go to a placeholder. The current assignment stays optimal
    there, with value V + rho m, and this dual is feasible for the panel's own dual (its prices are nonnegative wherever a
    placeholder is still unused), so by weak duality
        Delta(r | S) <= rho m + TopSum_{b_r} [a_qr - pi_q]_+   for every rho >= 0.
    rho = 0 gives prices at least as large as tight_pi's. A larger rho charges for the free placeholder slots once, in
    the term rho m, where the plain bound lets each of the reviewer's b_r units use them."""
    qasg, qcnt, asg, load, phiQ, phiW, phiT = panel.base
    _dist_from_sink_priced(
        panel.nQ,
        panel.nW,
        panel.A,
        qasg,
        qcnt,
        asg,
        load,
        phiQ,
        phiW,
        phiT,
        panel.dist,
        panel.pred,
        panel.hn,
        panel.hpos,
        panel.settled,
        panel.mark,
        n_dummy,
        np.int64(rho),
    )
    dq = panel.dist[: panel.nQ]
    return np.where(dq < fl.INF, dq + phiQ - phiT[0], np.int64(4 * fl.SCALE))


RHO_GRID = tuple(int(round(fl.SCALE * j / 16)) for j in range(1, 17))


# ---------------------------------------------------------------------------------------------------- dual bounds
@njit(cache=True, parallel=True)
def _dual_bounds_par(pi, AC, bC, active, out):
    """flow._dual_bounds with the candidates spread over the threads (the same integers)."""
    nC, nQ = AC.shape
    for c in prange(nC):
        if not active[c]:
            out[c] = -1
            continue
        b = bC[c]
        top = np.zeros(max(b, 1), np.int64)
        m = 0
        for q in range(nQ):
            a = AC[c, q]
            if a < 0:
                continue
            v = a - pi[q]
            if v <= 0:
                continue
            if m < b:
                j = m
                m += 1
            elif v > top[m - 1]:
                j = m - 1
            else:
                continue
            while j > 0 and top[j - 1] < v:
                top[j] = top[j - 1]
                j -= 1
            top[j] = v
        s = np.int64(0)
        for j in range(m):
            s += top[j]
        out[c] = s


def dual_bounds(pi, AC, bC, active, out):
    """U_r = TopSum_{b_r} [a_qr - pi_q]_+ for every active candidate: the single-threaded pass when Numba has one thread
    (timing runs), the threaded copy otherwise."""
    if get_num_threads() > 1:
        _dual_bounds_par(pi, AC, bC, active, out)
    else:
        fl._dual_bounds(pi, AC, bC, active, out)


# --------------------------------------------------------------------------------------------------------- greedy
def greedy(panel, AC, bC, k, mode="priced", ub_js=(), n_dummy=0, refine_after=16, lazy=True, dual=True, max_seconds=None):
    """Exact greedy recruitment of up to k candidates (rows of AC, capacities bC) into `panel`.

    mode selects the dual bounds:
      "potential"  prices read from the current potentials (Panel.pi)
      "tight"      tight_pi: one shortest-path pass from the sink per round
      "priced"     priced_pi at rho = 0 and, in a round that has evaluated refine_after candidates without closing
                   while placeholder reviews remain, the minimum over RHO_GRID of rho m + U_r(rho). Bounds of earlier
                   rounds carry over. This is CertGreedy as evaluated in the paper.
    lazy and dual switch the two certificates. Every setting selects the same recruits:
      lazy=True,  dual=True   CertGreedy
      lazy=True,  dual=False  Lazy Greedy: exact marginals of earlier rounds only, so round 1 evaluates every candidate
      lazy=False, dual=True   CertGreedy without lazy evaluation: the dual bounds of the current round only
      lazy=False, dual=False  Incremental-Flow Greedy: every candidate is evaluated in every round
    n_dummy is the number of placeholder reviewers, which are the first columns of the panel.
    ub_js: budgets j for which to record the a posteriori bound min_i [G(S_i) + TopSum_j(keys at S_i)] on the optimum.
    run.cum_seconds and run.cum_trials hold the clock and the number of evaluations after every round. max_seconds ends
    the run after the round in which the clock passes it (run.timed_out)."""
    nC = AC.shape[0]
    bC = np.asarray(bC, np.int64)
    active = fl.usable(AC, bC)
    stale = np.full(nC, fl.INF, dtype=np.int64)
    run = fl.Run(method=f"greedy-{mode}" + ("" if lazy else "-nolazy") + ("" if dual else "-nodual"))
    run.cum_seconds, run.cum_trials, run.timed_out = [], [], False
    run.refined_rounds = 0
    ub = {j: None for j in ub_js}
    d0, s0 = panel.stats.copy(), panel.n_trials
    bnd = np.zeros(nC, np.int64)
    tmp = np.zeros(nC, np.int64)
    idx = np.arange(nC)
    t_start = time.perf_counter()
    G = 0
    for i in range(k):
        t_b = time.perf_counter()
        if dual:
            if mode == "priced":
                pi = priced_pi(panel, n_dummy, 0)
            else:
                pi = tight_pi(panel) if mode == "tight" else panel.pi()
            dual_bounds(pi, AC, bC, active, bnd)
            if mode == "priced" and lazy:
                stale = np.minimum(stale, np.where(active, bnd, stale))
            keys = np.minimum(stale, bnd) if lazy else bnd.copy()
        else:
            keys = stale.copy() if lazy else np.full(nC, fl.INF, dtype=np.int64)
        keys[~active] = -1
        order = np.lexsort((idx, -keys))
        run.bound_seconds += time.perf_counter() - t_b
        best, best_c, n_eval = 0, -1, 0
        seen = np.zeros(nC, bool)
        m_free = int(panel.base[3][:n_dummy].sum()) if n_dummy else 0
        refined = False
        j = 0
        while j < nC:
            c = order[j]
            j += 1
            if seen[c]:
                continue
            kc = keys[c]
            if kc < 0:
                break
            if kc < best or (kc == best and (best_c < 0 or c > best_c)):
                break
            if dual and mode == "priced" and m_free > 0 and not refined and n_eval >= refine_after:
                t_b = time.perf_counter()
                ref = np.full(nC, fl.INF, dtype=np.int64)
                for rho in RHO_GRID:
                    dual_bounds(priced_pi(panel, n_dummy, rho), AC, bC, active, tmp)
                    np.minimum(ref, tmp + np.int64(rho) * m_free, out=ref)
                upd = active & ~seen
                stale[upd] = np.minimum(stale[upd], ref[upd])
                keys[upd] = np.minimum(keys[upd], ref[upd])
                order = np.lexsort((idx, -keys))
                refined = True
                run.refined_rounds += 1
                run.bound_seconds += time.perf_counter() - t_b
                j = 0
                continue
            g = panel.trial(AC[c], bC[c])
            n_eval += 1
            seen[c] = True
            assert g <= kc, "a certified bound is below an exact marginal"
            keys[c] = g
            stale[c] = g
            if g > best or (g == best and g > 0 and c < best_c):
                best, best_c = g, c
                panel.keep_trial_as_best(AC[c], bC[c])
        run.per_round_trials.append(n_eval)
        if ub:
            ka = np.where(active, keys, 0)
            srt = np.sort(ka)[::-1]
            for j in ub:
                v = G + int(srt[:j].sum())
                ub[j] = v if ub[j] is None else min(ub[j], v)
        if best <= 0:
            run.stopped_early = True
            break
        panel.commit_best(("c", int(best_c)))
        active[best_c] = False
        stale[best_c] = -1
        run.selected.append(int(best_c))
        run.gains.append(int(best))
        G += best
        run.cum_seconds.append(time.perf_counter() - t_start)
        run.cum_trials.append(int(panel.n_trials - s0))
        if max_seconds is not None and run.cum_seconds[-1] > max_seconds:
            run.timed_out = True
            break
    run.seconds = time.perf_counter() - t_start
    run.trials = panel.n_trials - s0
    run.dijkstra, run.settled = (panel.stats - d0).tolist()
    run.ub = ub
    return run


# The exact-marginal greedy variants of the experiments, as keyword arguments of `greedy`.
VARIANTS = {
    "certgreedy": {"lazy": True, "dual": True},
    "lazy": {"lazy": True, "dual": False},  # Lazy Greedy
    "dual_only": {"lazy": False, "dual": True},  # CertGreedy without lazy evaluation
    "incremental_flow": {"lazy": False, "dual": False},  # Incremental-Flow Greedy
}
