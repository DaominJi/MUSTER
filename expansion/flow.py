"""Exact reviewer assignment as a min-cost flow, with incremental insertion of one reviewer (InsertReviewer).

Network: source s -> submission q (capacity d_q, cost 0), q -> reviewer w (capacity 1, cost -a_qw) for every eligible
pair, reviewer w -> sink t (capacity b_w, cost 0). A min-cost flow of value T = sum_q d_q is an optimal assignment, and
its cost is minus the assignment value V_a(W).

Affinities are quantized to integer micro-units (a_int = round(a * 1e6)), so flows, potentials, gains and bounds are
exact integers and every comparison between algorithms is exact. Ineligible pairs (conflicts of interest) are -1.

Potentials phi keep every residual reduced cost c(u, v) + phi(u) - phi(v) nonnegative. The dual of the assignment LP is
read from them: pi_q = phi(q) - phi(t) and nu_w = [phi(t) - phi(w)]_+.

The source is never on an insertion path (every arc s -> q is saturated), so the insertion works on {t, Q, W, r}.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

INF = np.int64(1 << 62)
SCALE = 1_000_000  # micro-units per unit of affinity


def quantize(a):
    """Affinities in [0, 1] (NaN or negative = ineligible) -> int64 micro-units, -1 = ineligible."""
    a = np.asarray(a, dtype=np.float64)
    bad = ~np.isfinite(a) | (a < 0)
    out = np.rint(np.clip(np.where(bad, 0.0, a), 0.0, 1.0) * SCALE).astype(np.int64)
    out[bad] = -1
    return out


# ---------------------------------------------------------------------------------------------------------- heap
@njit(cache=True)
def _hpush(hn, hpos, key, size, node):
    i = size
    k = key[node]
    while i > 0:
        p = (i - 1) >> 1
        pn = hn[p]
        if key[pn] <= k:
            break
        hn[i] = pn
        hpos[pn] = i
        i = p
    hn[i] = node
    hpos[node] = i
    return size + 1


@njit(cache=True)
def _hdecrease(hn, hpos, key, node):
    i = hpos[node]
    k = key[node]
    while i > 0:
        p = (i - 1) >> 1
        pn = hn[p]
        if key[pn] <= k:
            break
        hn[i] = pn
        hpos[pn] = i
        i = p
    hn[i] = node
    hpos[node] = i


@njit(cache=True)
def _hpop(hn, hpos, key, size):
    top = hn[0]
    hpos[top] = -1
    size -= 1
    if size > 0:
        node = hn[size]
        k = key[node]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= size:
                break
            c = l
            r = l + 1
            if r < size and key[hn[r]] < key[hn[l]]:
                c = r
            if key[hn[c]] >= k:
                break
            hn[i] = hn[c]
            hpos[hn[i]] = i
            i = c
        hn[i] = node
        hpos[node] = i
    return top, size


@njit(cache=True)
def _relax(v, nd, dist, pred, u, hn, hpos, size):
    if nd < dist[v]:
        dist[v] = nd
        pred[v] = u
        if hpos[v] >= 0:
            _hdecrease(hn, hpos, dist, v)
        else:
            size = _hpush(hn, hpos, dist, size, v)
    return size


# ----------------------------------------------------------------------------------------------- assignment lists
@njit(cache=True)
def _assign(q, w, qasg, qcnt, asg, load):
    qasg[q, qcnt[q]] = w
    qcnt[q] += 1
    asg[w, load[w]] = q
    load[w] += 1


@njit(cache=True)
def _unassign(q, w, qasg, qcnt, asg, load):
    n = qcnt[q]
    for j in range(n):
        if qasg[q, j] == w:
            qasg[q, j] = qasg[q, n - 1]
            break
    qcnt[q] = n - 1
    n = load[w]
    for j in range(n):
        if asg[w, j] == q:
            asg[w, j] = asg[w, n - 1]
            break
    load[w] = n - 1


# ------------------------------------------------------------------------------------------------- initial solve
@njit(cache=True)
def _ssp_initial(nQ, nW, A, d, cap, qasg, qcnt, asg, load, phiQ, phiW, phiT, dist, pred, hn, hpos, settled, mark, flow_s):
    """Successive shortest paths from s on reduced costs. Returns (#augmentations, #Dijkstra settles) or -1 if infeasible.
    Node ids: q in [0, nQ), reviewer w at nQ + w, sink t at nQ + nW. pred = -1 marks the source s."""
    T = 0
    for q in range(nQ):
        T += d[q]
    nN = nQ + nW + 1
    tN = nQ + nW
    phiS = np.int64(0)
    work = np.int64(0)
    for it in range(T):
        for v in range(nN):
            dist[v] = INF
            pred[v] = -2
            hpos[v] = -1
            settled[v] = 0
        size = 0
        for q in range(nQ):
            if flow_s[q] < d[q]:
                size = _relax(q, phiS - phiQ[q], dist, pred, -1, hn, hpos, size)
        found = False
        while size > 0:
            u, size = _hpop(hn, hpos, dist, size)
            settled[u] = 1
            work += 1
            du = dist[u]
            if u == tN:
                found = True
                break
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
                            size = _relax(v, du - aw + pq - phiW[w], dist, pred, u, hn, hpos, size)
                for j in range(qcnt[q]):
                    mark[qasg[q, j]] = 0
            else:
                w = u - nQ
                pw = phiW[w]
                for j in range(load[w]):
                    q = asg[w, j]
                    if settled[q] == 0:
                        size = _relax(q, du + A[q, w] + pw - phiQ[q], dist, pred, u, hn, hpos, size)
                if load[w] < cap[w] and settled[tN] == 0:
                    size = _relax(tN, du + pw - phiT[0], dist, pred, u, hn, hpos, size)
        if not found:
            return -1, work
        K = dist[tN]
        for q in range(nQ):
            dq = dist[q]
            phiQ[q] += dq if dq < K else K
        for w in range(nW):
            dw = dist[nQ + w]
            phiW[w] += dw if dw < K else K
        phiT[0] += K
        # augment s -> ... -> t
        v = tN
        while True:
            u = pred[v]
            if u == -1:
                flow_s[v] += 1
                break
            if v == tN:
                pass  # arc w -> t: the load was already counted by the incoming q -> w arc
            elif v >= nQ:
                _assign(u, v - nQ, qasg, qcnt, asg, load)
            else:
                _unassign(v, u - nQ, qasg, qcnt, asg, load)
            v = u
    return T, work


# ----------------------------------------------------------------------------------------------------- insertion
@njit(cache=True)
def _insert(nQ, nW, A, b_r, qasg, qcnt, asg, load, phiQ, phiW, phiT, dist, pred, hn, hpos, settled, mark, stats):
    """InsertReviewer for the trial reviewer stored in column nW of A, on the given (working) state.

    The state arrays are modified in place and end as an optimal flow, with valid potentials, for the panel plus
    the trial reviewer (index nW, capacity b_r). Returns the exact marginal gain. stats[0] += Dijkstra runs,
    stats[1] += settled nodes."""
    R = nW  # reviewer index of the trial candidate
    RN = nQ + nW  # its node id
    phi_r = INF
    for q in range(nQ):
        a = A[q, R]
        if a >= 0:
            v = phiQ[q] - a
            if v < phi_r:
                phi_r = v
    load[R] = 0
    if phi_r == INF or b_r == 0:
        # no capacity or no eligible submission: the marginal is zero; phi_r keeps arcs q -> r valid
        phiW[R] = phiT[0] if phi_r == INF else phi_r
        return np.int64(0)
    phiW[R] = phi_r
    nN = nQ + nW + 1
    gain = np.int64(0)
    for it in range(b_r):
        stats[0] += 1
        for v in range(nN):
            dist[v] = INF
            pred[v] = -2
            hpos[v] = -1
            settled[v] = 0
        size = 0
        for w in range(nW + 1):
            if load[w] > 0:
                size = _relax(nQ + w, phiT[0] - phiW[w], dist, pred, -1, hn, hpos, size)
        found = False
        while size > 0:
            u, size = _hpop(hn, hpos, dist, size)
            settled[u] = 1
            stats[1] += 1
            du = dist[u]
            if u == RN:
                found = True
                break
            if u < nQ:
                q = u
                pq = phiQ[q]
                for j in range(qcnt[q]):
                    mark[qasg[q, j]] = 1
                for w in range(nW + 1):
                    aw = A[q, w]
                    if aw >= 0 and mark[w] == 0:
                        v = nQ + w
                        if settled[v] == 0:
                            size = _relax(v, du - aw + pq - phiW[w], dist, pred, u, hn, hpos, size)
                for j in range(qcnt[q]):
                    mark[qasg[q, j]] = 0
            else:
                w = u - nQ
                pw = phiW[w]
                for j in range(load[w]):
                    q = asg[w, j]
                    if settled[q] == 0:
                        size = _relax(q, du + A[q, w] + pw - phiQ[q], dist, pred, u, hn, hpos, size)
        if found:
            K = dist[RN]
            chi = dist[RN] - phiT[0] + phiW[R]
        else:
            K = phiT[0] - phiW[R]
            if K < 0:
                K = 0
            chi = INF
        for q in range(nQ):
            dq = dist[q]
            phiQ[q] += dq if dq < K else K
        for w in range(nW + 1):
            dw = dist[nQ + w]
            phiW[w] += dw if dw < K else K
        if chi >= 0:
            break
        v = RN
        while True:
            u = pred[v]
            if u == -1:
                break
            if v >= nQ:
                _assign(u, v - nQ, qasg, qcnt, asg, load)
            else:
                _unassign(v, u - nQ, qasg, qcnt, asg, load)
            v = u
        gain -= chi
    return gain


@njit(cache=True)
def _copy_state(nQ, nW1, qasg, qcnt, asg, load, phiQ, phiW, phiT, qasg2, qcnt2, asg2, load2, phiQ2, phiW2, phiT2):
    for q in range(nQ):
        qcnt2[q] = qcnt[q]
        phiQ2[q] = phiQ[q]
        for j in range(qcnt[q]):
            qasg2[q, j] = qasg[q, j]
    for w in range(nW1):
        load2[w] = load[w]
        phiW2[w] = phiW[w]
        for j in range(load[w]):
            asg2[w, j] = asg[w, j]
    phiT2[0] = phiT[0]


@njit(cache=True)
def _trial(
    nQ,
    nW,
    A,
    a_r,
    b_r,
    B_qasg,
    B_qcnt,
    B_asg,
    B_load,
    B_phiQ,
    B_phiW,
    B_phiT,
    qasg,
    qcnt,
    asg,
    load,
    phiQ,
    phiW,
    phiT,
    dist,
    pred,
    hn,
    hpos,
    settled,
    mark,
    stats,
):
    """Copy the base state into the trial buffers, place a_r in column nW, and run the insertion."""
    _copy_state(nQ, nW, B_qasg, B_qcnt, B_asg, B_load, B_phiQ, B_phiW, B_phiT, qasg, qcnt, asg, load, phiQ, phiW, phiT)
    for q in range(nQ):
        A[q, nW] = a_r[q]
    return _insert(nQ, nW, A, b_r, qasg, qcnt, asg, load, phiQ, phiW, phiT, dist, pred, hn, hpos, settled, mark, stats)


@njit(cache=True)
def _dual_bounds(pi, AC, bC, active, out):
    """U_r = TopSum_{b_r} [a_qr - pi_q]_+ over the eligible submissions q of every active candidate r (rows of AC):
    the dual upper bound on the marginal gain of r at the submission prices pi."""
    nC, nQ = AC.shape
    top = np.zeros(64, dtype=np.int64)
    for c in range(nC):
        if not active[c]:
            out[c] = -1
            continue
        b = bC[c]
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


@njit(cache=True)
def _check_potentials(nQ, nW, A, cap, qasg, qcnt, asg, load, phiQ, phiW, phiT, mark):
    """Minimum reduced cost over all residual arcs on {Q, W, t} (must be >= 0 for an optimal flow)."""
    worst = INF
    for q in range(nQ):
        for j in range(qcnt[q]):
            mark[qasg[q, j]] = 1
        for w in range(nW):
            a = A[q, w]
            if a < 0:
                continue
            if mark[w] == 1:
                rc = a + phiW[w] - phiQ[q]  # reverse arc w -> q
            else:
                rc = -a + phiQ[q] - phiW[w]  # forward arc q -> w
            if rc < worst:
                worst = rc
        for j in range(qcnt[q]):
            mark[qasg[q, j]] = 0
    for w in range(nW):
        if load[w] > 0:
            rc = phiT[0] - phiW[w]
            if rc < worst:
                worst = rc
        if load[w] < cap[w]:
            rc = phiW[w] - phiT[0]
            if rc < worst:
                worst = rc
    return worst


# ------------------------------------------------------------------------------------------------- panel state
class Panel:
    """An optimal assignment for a growing panel, with potentials, supporting exact insertion trials.

    A0: (nQ, n0) int64 incumbent affinities (-1 = ineligible), d: (nQ,) demands, b0: (n0,) capacities.
    The panel can grow by up to `extra` reviewers (the budget)."""

    def __init__(self, A0, d, b0, extra, bmax=None):
        A0 = np.asarray(A0, dtype=np.int64)
        self.nQ, self.n0 = A0.shape
        self.d = np.asarray(d, dtype=np.int32)
        self.nWmax = self.n0 + int(extra) + 1
        self.dmax = int(self.d.max())
        self.bmax = int(max(int(np.max(b0)), bmax or 0))
        nQ, Wm = self.nQ, self.nWmax
        self.A = np.full((nQ, Wm), -1, dtype=np.int64)
        self.A[:, : self.n0] = A0
        self.cap = np.zeros(Wm, dtype=np.int32)
        self.cap[: self.n0] = np.asarray(b0, dtype=np.int32)
        self.nW = self.n0
        self.members = list(range(self.n0))  # labels: incumbents 0..n0-1, recruits as ("c", index)

        def bufs():
            return [
                np.zeros((nQ, self.dmax + 1), np.int32),
                np.zeros(nQ, np.int32),
                np.zeros((Wm, self.bmax + 1), np.int32),
                np.zeros(Wm, np.int32),
                np.zeros(nQ, np.int64),
                np.zeros(Wm, np.int64),
                np.zeros(1, np.int64),
            ]

        self.base, self.tr, self.best = bufs(), bufs(), bufs()
        nN = nQ + Wm + 1
        self.dist = np.zeros(nN, np.int64)
        self.pred = np.zeros(nN, np.int64)
        self.hn = np.zeros(nN, np.int64)
        self.hpos = np.zeros(nN, np.int64)
        self.settled = np.zeros(nN, np.int8)
        self.mark = np.zeros(Wm, np.int8)
        self.stats = np.zeros(2, np.int64)  # Dijkstra runs, settled nodes (insertion only)
        self.value = None
        self.n_trials = 0

    # -- initial solve
    def solve(self):
        qasg, qcnt, asg, load, phiQ, phiW, phiT = self.base
        phiQ[:] = 0
        phiW[:] = -SCALE
        phiT[0] = -SCALE
        flow_s = np.zeros(self.nQ, np.int32)
        t0 = time.perf_counter()
        T, work = _ssp_initial(
            self.nQ,
            self.nW,
            self.A,
            self.d,
            self.cap,
            qasg,
            qcnt,
            asg,
            load,
            phiQ,
            phiW,
            phiT,
            self.dist,
            self.pred,
            self.hn,
            self.hpos,
            self.settled,
            self.mark,
            flow_s,
        )
        self.solve_time = time.perf_counter() - t0
        if T < 0:
            raise ValueError("incumbent panel is infeasible")
        self.value = self.assignment_value()
        return self.value

    def assignment_value(self, which="base"):
        qasg, qcnt = (self.base if which == "base" else self.best)[:2]
        v = 0
        for q in range(self.nQ):
            for j in range(qcnt[q]):
                v += int(self.A[q, qasg[q, j]])
        return v

    def pi(self):
        """Submission prices pi_q = phi(q) - phi(t) of the current optimal dual."""
        return self.base[4] - self.base[6][0]

    def dual_objective(self):
        """sum_q d_q pi_q + sum_w b_w nu_w + sum zeta (equals the panel value at an optimal dual)."""
        phiQ, phiW, phiT = self.base[4], self.base[5], self.base[6][0]
        nW = self.nW
        pi = phiQ - phiT
        nu = np.maximum(phiT - phiW[:nW], 0)
        A = self.A[:, :nW]
        el = A >= 0
        zeta = np.where(el, np.maximum(A - pi[:, None] - nu[None, :], 0), 0)
        return int((self.d * pi).sum() + (self.cap[:nW] * nu).sum() + zeta.sum())

    def check(self):
        qasg, qcnt, asg, load, phiQ, phiW, phiT = self.base
        worst = _check_potentials(self.nQ, self.nW, self.A, self.cap, qasg, qcnt, asg, load, phiQ, phiW, phiT, self.mark)
        assert (qcnt == self.d).all(), "demand violated"
        assert (load[: self.nW] <= self.cap[: self.nW]).all(), "capacity violated"
        return int(worst)

    def check_trial(self):
        """Validity of the last trial flow (panel plus trial reviewer): min residual reduced cost."""
        qasg, qcnt, asg, load, phiQ, phiW, phiT = self.tr
        cap = self.cap.copy()
        cap[self.nW] = self._last_b
        worst = _check_potentials(self.nQ, self.nW + 1, self.A, cap, qasg, qcnt, asg, load, phiQ, phiW, phiT, self.mark)
        assert (qcnt == self.d).all() and (load[: self.nW + 1] <= cap[: self.nW + 1]).all()
        return int(worst)

    # -- insertion trials
    def trial(self, a_r, b_r):
        """Exact marginal gain of adding a reviewer with affinities a_r (int64, -1 ineligible) and capacity b_r.
        The trial flow stays in the trial buffers until `keep_trial_as_best` or the next trial."""
        b_r = int(min(b_r, int((a_r >= 0).sum())))
        assert b_r <= self.bmax, "candidate capacity exceeds the panel buffers (pass bmax)"
        self.n_trials += 1
        B, Tr = self.base, self.tr
        g = _trial(
            self.nQ, self.nW, self.A, a_r, b_r, *B, *Tr, self.dist, self.pred, self.hn, self.hpos, self.settled, self.mark, self.stats
        )
        self._last_b = b_r
        return int(g)

    def keep_trial_as_best(self, a_r, b_r):
        _copy_state(self.nQ, self.nW + 1, *self.tr, *self.best)
        self._best_a = np.array(a_r, copy=True)
        self._best_b = self._last_b

    def commit_best(self, label):
        """Make the kept best trial flow the new base (the trial reviewer joins the panel)."""
        nW = self.nW
        self.A[:, nW] = self._best_a
        self.cap[nW] = self._best_b
        _copy_state(self.nQ, nW + 1, *self.best, *self.base)
        self.nW = nW + 1
        self.members.append(label)
        self.value = self.assignment_value()

    def add(self, a_r, b_r, label):
        """Insert a reviewer and commit (exact). Returns its marginal gain."""
        g = self.trial(a_r, b_r)
        self.keep_trial_as_best(a_r, b_r)
        self.commit_best(label)
        return g

    def assignment(self):
        """List of (q, member label) pairs of the current optimal assignment."""
        qasg, qcnt = self.base[:2]
        return [(q, self.members[qasg[q, j]]) for q in range(self.nQ) for j in range(qcnt[q])]


# --------------------------------------------------------------------------------------------------- recruitment
@dataclass
class Run:
    method: str
    selected: list = field(default_factory=list)
    gains: list = field(default_factory=list)  # marginal gain of each recruit (exact)
    trials: int = 0
    dijkstra: int = 0
    settled: int = 0
    seconds: float = 0.0
    bound_seconds: float = 0.0
    ub: dict = field(default_factory=dict)  # j -> certified upper bound on G(S*_j)
    per_round_trials: list = field(default_factory=list)
    stopped_early: bool = False

    def cum(self):
        return np.cumsum([0] + list(self.gains))


def usable(AC, bC):
    """Candidates with positive capacity and at least one eligible submission."""
    return (np.asarray(bC) > 0) & ((AC >= 0).sum(axis=1) > 0)


def evaluate_sequence(panel, AC, bC, order, k, method, checkpoints=(), on_checkpoint=None):
    """Recruit the first k candidates of a fixed ranking, reoptimizing the assignment after each (exact).
    on_checkpoint(j, panel) is called after j recruits for every j in checkpoints (and for j = 0)."""
    run = Run(method=method)
    t0 = time.perf_counter()
    if on_checkpoint is not None and 0 in checkpoints:
        on_checkpoint(0, panel)
    n_done = 0
    for j, c in enumerate(order[:k], start=1):
        g = panel.add(AC[c], bC[c], ("c", int(c)))
        run.selected.append(int(c))
        run.gains.append(int(g))
        n_done = j
        if on_checkpoint is not None and j in checkpoints:
            on_checkpoint(j, panel)
    if on_checkpoint is not None:  # a shorter order (greedy stopped early: no candidate can help) keeps its final state
        for j in sorted(checkpoints):
            if j > n_done:
                on_checkpoint(j, panel)
    run.seconds = time.perf_counter() - t0
    return run


# -------------------------------------------------------------------------------------------- from-scratch solve
def scratch_value(A_cols, d, b_cols):
    """Optimal assignment value from scratch with OR-Tools min-cost flow (integer costs).

    A_cols: (nQ, n) int64 affinities (-1 ineligible), b_cols: (n,) capacities. Returns value or None if infeasible."""
    from ortools.graph.python import min_cost_flow

    nQ, n = A_cols.shape
    s, t = nQ + n, nQ + n + 1
    qi, wi = np.nonzero(A_cols >= 0)
    tails = np.concatenate([np.full(nQ, s), qi, nQ + np.arange(n)])
    heads = np.concatenate([np.arange(nQ), nQ + wi, np.full(n, t)])
    caps = np.concatenate([np.asarray(d, np.int64), np.ones(len(qi), np.int64), np.asarray(b_cols, np.int64)])
    costs = np.concatenate([np.zeros(nQ, np.int64), -A_cols[qi, wi], np.zeros(n, np.int64)])
    smcf = min_cost_flow.SimpleMinCostFlow()
    smcf.add_arcs_with_capacity_and_unit_cost(tails.astype(np.int32), heads.astype(np.int32), caps, costs)
    T = int(np.sum(d))
    sup = np.zeros(nQ + n + 2, np.int64)
    sup[s], sup[t] = T, -T
    smcf.set_nodes_supplies(np.arange(nQ + n + 2, dtype=np.int32), sup)
    st = smcf.solve()
    if st != smcf.OPTIMAL:
        return None
    return int(-smcf.optimal_cost())
