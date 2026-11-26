"""Exact optimum of budgeted panel expansion.

`mip_optimum` solves the mixed-integer program of the technical report ("Exact Mixed-Integer Reference") over all
candidates with HiGHS. It suits instances with up to a few thousand candidates.

`sift` solves full venue instances by column generation over the candidates and the assignment edges.

Bound. For prices pi (one per submission, any sign) and mu >= 0,
    UB(pi, mu) = sum_q d_q pi_q + sum_{reviewers w of the panel} T_w(pi) + k mu + sum_{candidates r} [T_r(pi) - mu]_+,
    T_w(pi) = sum of the b_w largest positive a_qw - pi_q over the eligible submissions of w,
bounds the optimum of the MIP. Two consequences, with L the value of a known selection, t_w the b_w-th largest positive
a_qw - pi_q of reviewer w (0 if there are fewer), and integer values:
  (1) a selection that contains candidate r is worth at most UB + min(T_r - mu, 0);
  (2) an assignment that uses the edge (q, w) is worth at most UB - max(0, t_w - (a_qw - pi_q)), and for an edge of a
      candidate the term of (1) comes on top.
So every selection worth L + 1 or more uses only the candidates and edges whose bound in (1) and (2) is at least L + 1,
and the MIP on these candidates and edges alone decides the optimum.

Prices. The LP relaxation is solved on a working set of candidates and edges. At its prices, every panel member and
every working candidate must have its b_w best positive edges in the edge set, and no outside candidate may have
T_r > mu. Otherwise the missing edges and candidates are added and the LP is solved again. The LP phase ends when
nothing is missing, or when UB(pi, mu) equals the LP value of the working sets (no prices can give a smaller bound).
UB is valid for any prices, so the result does not depend on how the prices were found.

Deciding MIP. It carries the row "value >= L + 1/2", so it is infeasible exactly when L is the optimum. When HiGHS
reports it solved without a better selection (the row was met only within the solver's tolerances), the same MIP is
solved again without the row. The value of every selection is recomputed exactly with a from-scratch assignment.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit

from . import flow as fl

PAD = 6  # edges kept per reviewer beyond its capacity when the edge set grows
HEUR = 0.5  # share of the remaining time that the MIP on the working sets may take


# ------------------------------------------------------------------------------------------- MIP over all candidates
def mip_optimum(A0, b0, AC, bC, d, k, time_limit=600.0, mip_gap=0.0, strong=False):
    """The exact MIP over all candidates with HiGHS (scipy.optimize.milp).

    x_qr in [0, 1] continuous (the assignment polytope is integral for fixed y), y_r binary for candidates.
    strong=True adds the valid inequalities x_qr <= y_r on candidate edges (same optimum, tighter relaxation).
    Returns dict(value, bound, status, selected, seconds)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    nQ, n0 = A0.shape
    nC = AC.shape[0]
    A_all = np.hstack([A0, AC.T])  # (nQ, n0 + nC)
    qi, wi = np.nonzero(A_all >= 0)
    nx = len(qi)
    ny = nC
    c = np.concatenate([-A_all[qi, wi].astype(np.float64), np.zeros(ny)])
    rows, cols, vals = [], [], []
    # demand rows
    rows.append(qi)
    cols.append(np.arange(nx))
    vals.append(np.ones(nx))
    # capacity rows (one per reviewer): sum_q x_qw - b_w y_w <= 0 (candidates), <= b_w (incumbents)
    rows.append(nQ + wi)
    cols.append(np.arange(nx))
    vals.append(np.ones(nx))
    cand = np.arange(nC)
    rows.append(nQ + n0 + cand)
    cols.append(nx + cand)
    vals.append(-np.asarray(bC, np.float64))
    # budget row
    rows.append(np.full(ny, nQ + n0 + nC))
    cols.append(nx + cand)
    vals.append(np.ones(ny))
    n_rows = nQ + n0 + nC + 1
    lb = [np.asarray(d, np.float64), np.full(n0 + nC, -np.inf), [-np.inf]]
    ub = [np.asarray(d, np.float64), np.asarray(b0, np.float64), np.zeros(nC), [float(k)]]
    if strong:  # x_qr - y_r <= 0 for every candidate edge
        ce = np.flatnonzero(wi >= n0)
        rr = n_rows + np.arange(len(ce))
        rows += [rr, rr]
        cols += [ce, nx + (wi[ce] - n0)]
        vals += [np.ones(len(ce)), -np.ones(len(ce))]
        n_rows += len(ce)
        lb.append(np.full(len(ce), -np.inf))
        ub.append(np.zeros(len(ce)))
    M = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_rows, nx + ny)).tocsr()
    lb, ub = np.concatenate(lb), np.concatenate(ub)
    integrality = np.concatenate([np.zeros(nx), np.ones(ny)])
    t0 = time.perf_counter()
    res = milp(
        c,
        constraints=LinearConstraint(M, lb, ub),
        integrality=integrality,
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": False},
    )
    sec = time.perf_counter() - t0
    out = {"status": int(res.status), "message": res.message, "seconds": sec}
    if res.x is not None:
        y = res.x[nx:]
        out["selected"] = [int(i) for i in np.flatnonzero(y > 0.5)]
        out["value"] = float(-res.fun)
        out["bound"] = (
            float(-getattr(res, "mip_dual_bound", res.fun)) if getattr(res, "mip_dual_bound", None) is not None else float(-res.fun)
        )
    return out


# ------------------------------------------------------------------------------------------------------- sifting
@njit(cache=True)
def float_bounds(pi, AC, bC, out):
    """U_r(pi) = sum of the b_r largest positive a_qr - pi_q over the eligible submissions of every row of AC."""
    nC, nQ = AC.shape
    for c in range(nC):
        b = bC[c]
        top = np.zeros(max(b, 1), np.float64)
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
        s = 0.0
        for j in range(m):
            s += top[j]
        out[c] = s


@njit(cache=True)
def row_top(pi, M, bvec, K, out_q, out_v, out_T, out_t):
    """For every row w of M (reviewers x submissions, -1 = ineligible): the K eligible submissions with the largest
    v = M[w, q] - pi[q] (out_q, -1 padded, with out_v), T_w = the sum of the b_w largest positive v, and t_w = the
    b_w-th largest positive v (0 if fewer are positive)."""
    nW, nQ = M.shape
    for w in range(nW):
        m = 0
        for q in range(nQ):
            a = M[w, q]
            if a < 0:
                continue
            v = a - pi[q]
            if m < K:
                j = m
                m += 1
            elif v > out_v[w, m - 1]:
                j = m - 1
            else:
                continue
            while j > 0 and out_v[w, j - 1] < v:
                out_v[w, j] = out_v[w, j - 1]
                out_q[w, j] = out_q[w, j - 1]
                j -= 1
            out_v[w, j] = v
            out_q[w, j] = q
        for j in range(m, K):
            out_q[w, j] = -1
            out_v[w, j] = -1e300
        b = bvec[w]
        T, t = 0.0, 0.0
        for j in range(min(b, m)):
            if out_v[w, j] > 0:
                T += out_v[w, j]
                if j == b - 1:
                    t = out_v[w, j]
        out_T[w] = T
        out_t[w] = t


@njit(cache=True)
def edges_above(pi, M, thr, cap):
    """The eligible edges (w, q) of M with M[w, q] - pi[q] >= thr[w], as two arrays (at most cap, then the count is -1)."""
    nW, nQ = M.shape
    ws = np.empty(cap, np.int64)
    qs = np.empty(cap, np.int64)
    n = 0
    for w in range(nW):
        th = thr[w]
        for q in range(nQ):
            a = M[w, q]
            if a < 0:
                continue
            if a - pi[q] >= th:
                if n >= cap:
                    return ws, qs, -1
                ws[n] = w
                qs[n] = q
                n += 1
    return ws, qs, n


class Problem:
    """One instance at one budget k: incumbent affinities A0 (submissions x incumbents), candidates AC (rows)."""

    def __init__(self, A0, b0, AC, bC, d, k):
        self.A0, self.b0, self.AC, self.bC, self.d, self.k = (
            A0,
            np.asarray(b0, np.int64),
            AC,
            np.asarray(bC, np.int64),
            np.asarray(d, np.int64),
            k,
        )
        self.nQ, self.n0 = A0.shape
        self.nC = AC.shape[0]
        self.free = self.b0 >= self.nQ  # placeholders: zero affinity, eligible for everything, unlimited capacity
        self.inc = np.flatnonzero(~self.free)
        self.A0T = np.ascontiguousarray(A0.T[self.inc]).astype(np.int64)  # incumbents x submissions
        self.act = fl.usable(AC, self.bC)
        self.K = int(max(self.b0[self.inc].max() if len(self.inc) else 0, self.bC.max())) + PAD

    def solve(self, cand, e_inc, e_cand, integer, limit=None, cutoff=None):
        """LP relaxation (or MIP) on the candidates `cand` with the incumbent edges e_inc (rows of A0T, q) and the candidate
        edges e_cand (position in cand, q). Placeholders keep all their edges. Returns value, pi, mu, selected candidates."""
        from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        from scipy.sparse import coo_matrix

        nQ = self.nQ
        ph = np.flatnonzero(self.free)
        n_ph, n_i, n_c = len(ph) * nQ, len(e_inc[0]), len(e_cand[0])
        nx = n_ph + n_i + n_c
        ny = len(cand)
        q_all = np.concatenate([np.tile(np.arange(nQ), len(ph)), e_inc[1], e_cand[1]])
        aff = np.concatenate(
            [np.zeros(n_ph), self.A0T[e_inc[0], e_inc[1]].astype(np.float64), self.AC[cand[e_cand[0]], e_cand[1]].astype(np.float64)]
        )
        c = np.concatenate([-aff, np.zeros(ny)])
        # rows: capacity of every incumbent, capacity of every working candidate, budget, one strong row per candidate edge
        r_i = e_inc[0]
        r_c = len(self.inc) + e_cand[0]
        r_bud = len(self.inc) + ny
        rr = r_bud + 1 + np.arange(n_c)
        xi, xc = n_ph + np.arange(n_i), n_ph + n_i + np.arange(n_c)
        rows = np.concatenate([r_i, r_c, len(self.inc) + np.arange(ny), np.full(ny, r_bud), rr, rr])
        cols = np.concatenate([xi, xc, nx + np.arange(ny), nx + np.arange(ny), xc, nx + e_cand[0]])
        vals = np.concatenate([np.ones(n_i), np.ones(n_c), -self.bC[cand].astype(np.float64), np.ones(ny), np.ones(n_c), -np.ones(n_c)])
        n_rows = r_bud + 1 + n_c
        A_ub = coo_matrix((vals, (rows, cols)), shape=(n_rows, nx + ny)).tocsr()
        b_ub = np.concatenate([self.b0[self.inc].astype(np.float64), np.zeros(ny), [float(self.k)], np.zeros(n_c)])
        A_eq = coo_matrix((np.ones(nx), (q_all, np.arange(nx))), shape=(nQ, nx + ny)).tocsr()
        b_eq = self.d.astype(np.float64)
        if not integer:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
            assert res.status == 0, res.message
            return -float(res.fun), -np.asarray(res.eqlin.marginals, np.float64), max(0.0, -float(res.ineqlin.marginals[r_bud])), None, 0
        cons = [LinearConstraint(A_ub, -np.inf, b_ub), LinearConstraint(A_eq, b_eq, b_eq)]
        if cutoff is not None:  # only assignments worth at least `cutoff` (in affinity units, so that the row is well scaled)
            nz = np.flatnonzero(aff)
            row = coo_matrix((aff[nz] * 1e-6, (np.zeros(len(nz), np.int64), nz)), shape=(1, nx + ny)).tocsr()
            cons.append(LinearConstraint(row, cutoff * 1e-6, np.inf))
        res = milp(
            c,
            constraints=cons,
            integrality=np.concatenate([np.zeros(nx), np.ones(ny)]),
            bounds=Bounds(0, 1),
            options={"time_limit": limit, "mip_rel_gap": 0.0, "disp": False},
        )
        sel = [int(cand[i]) for i in np.flatnonzero(res.x[nx:] > 0.5)] if res.x is not None else []
        return (-float(res.fun) if res.x is not None else None), None, None, sel, int(res.status)

    def value(self, sel):
        return int(
            fl.scratch_value(
                np.hstack([self.A0] + [self.AC[c][:, None].astype(np.int64) for c in sel]), self.d, np.concatenate([self.b0, self.bC[sel]])
            )
        )

    def bound(self, pi, mu, U):
        """UB(pi, mu) with the candidates' terms U, and the incumbents' terms (T, t) at these prices."""
        K = self.K
        oq, ov = np.empty((len(self.inc), K), np.int64), np.empty((len(self.inc), K), np.float64)
        T, t = np.zeros(len(self.inc)), np.zeros(len(self.inc))
        if len(self.inc):
            row_top(pi, self.A0T, self.b0[self.inc], K, oq, ov, T, t)
        ph = float(self.free.sum()) * float(np.maximum(-pi, 0.0).sum())
        ub = float(np.dot(self.d, pi) + T.sum() + ph + self.k * mu + np.maximum(U[self.act] - mu, 0.0).sum())
        return ub, oq, ov, T, t


def uniq(e):
    """Edges (w, q) without duplicates, sorted."""
    if len(e[0]) == 0:
        return e
    key = np.unique(e[0].astype(np.int64) * (1 << 32) + e[1].astype(np.int64))
    return key >> 32, key & ((1 << 32) - 1)


def sift(P, V0, g_sel, start_edges, limit, tol=0.5, edge_cap=6_000_000, lp_only=False, plain=False):
    """Optimum of the Problem `P` at its budget, starting from a known selection g_sel (CertGreedy's recruits).

    V0 is the value of the incumbent panel (for the progress lines), start_edges the (incumbent, submission) edges of
    its optimal assignment, and limit the time limit in seconds. Returns a dict with closed (True when the optimum is
    proven), value and selected (the best selection found), bound (the LP bound of the whole instance), rounds,
    seconds and mip. lp_only stops after the LP bound. plain leaves the row "value >= L + 1/2" out (a cross-check)."""
    t0 = time.perf_counter()
    K = P.K
    in_w = np.zeros(P.nC, bool)
    in_w[np.asarray(g_sel, np.int64)] = True
    e_inc = uniq(start_edges)  # the incumbent optimum's own edges keep the LP feasible
    cand_edges = {}  # candidate -> set of submissions
    L, best = P.value(list(g_sel)), list(g_sel)
    U = np.zeros(P.nC, np.float64)
    log = []
    pi = None
    for rnd in range(400):
        if time.perf_counter() - t0 > limit:
            return {
                "closed": False,
                "value": L,
                "selected": best,
                "bound": log[-1]["ub"] if log else None,
                "rounds": rnd,
                "log": log[-5:],
                "seconds": time.perf_counter() - t0,
            }
        cand = np.flatnonzero(in_w)
        if pi is None:  # first prices: the LP on the starting sets needs the candidates' best edges at zero prices
            pi0 = np.zeros(P.nQ)
            cq, cv = np.empty((len(cand), K), np.int64), np.empty((len(cand), K), np.float64)
            row_top(pi0, np.ascontiguousarray(P.AC[cand]).astype(np.int64), P.bC[cand], K, cq, cv, np.zeros(len(cand)), np.zeros(len(cand)))
            for i, r in enumerate(cand):
                cand_edges.setdefault(int(r), set()).update(int(q) for q in cq[i] if q >= 0)
        pos = {int(r): i for i, r in enumerate(cand)}
        ce_w = np.array([pos[r] for r in cand_edges if r in pos for _ in cand_edges[r]], np.int64)
        ce_q = np.array([q for r in cand_edges if r in pos for q in cand_edges[r]], np.int64)
        lp, pi, mu, _, _ = P.solve(cand, e_inc, (ce_w, ce_q), False)
        float_bounds(pi, P.AC, P.bC, U)
        ub, oq, ov, T, t = P.bound(pi, mu, U)
        # what the LP on the working sets misses at these prices
        new_inc = [[], []]
        have = set((e_inc[0] * (1 << 32) + e_inc[1]).tolist())
        for w in range(len(P.inc)):
            b = P.b0[P.inc[w]]
            for j in range(K):
                q = oq[w, j]
                if q < 0:
                    break
                if (j < b and ov[w, j] > 1e-6) or j < b + 2:
                    if (w << 32) + int(q) not in have:
                        if j < b and ov[w, j] > 1e-6:
                            new_inc[0].append(w)
                            new_inc[1].append(int(q))
        miss_inc = len(new_inc[0])
        if miss_inc:  # with the violated edges, also the neighbours in each row's top list
            ws = np.unique(np.asarray(new_inc[0], np.int64))
            add_w = np.repeat(ws, K)
            add_q = oq[ws].ravel()
            ok = add_q >= 0
            e_inc = uniq((np.concatenate([e_inc[0], add_w[ok]]), np.concatenate([e_inc[1], add_q[ok]])))
        cq, cv = np.empty((len(cand), K), np.int64), np.empty((len(cand), K), np.float64)
        row_top(pi, np.ascontiguousarray(P.AC[cand]).astype(np.int64), P.bC[cand], K, cq, cv, np.zeros(len(cand)), np.zeros(len(cand)))
        miss_c = 0
        for i, r in enumerate(cand):
            s = cand_edges[int(r)]
            b = P.bC[r]
            viol = any(cq[i, j] >= 0 and cv[i, j] > 1e-6 and int(cq[i, j]) not in s for j in range(min(b, K)))
            if viol:
                miss_c += 1
                s.update(int(q) for q in cq[i] if q >= 0)
        enter = P.act & ~in_w & (U - mu > 1e-6)
        n_ent = int(enter.sum())
        if n_ent:
            ent = np.flatnonzero(enter)
            if n_ent > 3000:  # the candidates with the largest reduced profit first
                ent = ent[np.argsort(-(U[ent] - mu))[:3000]]
            eq_, ev_ = np.empty((len(ent), K), np.int64), np.empty((len(ent), K), np.float64)
            row_top(pi, np.ascontiguousarray(P.AC[ent]).astype(np.int64), P.bC[ent], K, eq_, ev_, np.zeros(len(ent)), np.zeros(len(ent)))
            for i, r in enumerate(ent):
                cand_edges.setdefault(int(r), set()).update(int(q) for q in eq_[i] if q >= 0)
            in_w[ent] = True
        log.append(
            {
                "round": rnd,
                "candidates": int(len(cand)),
                "inc_edges": int(len(e_inc[0])),
                "lp": lp,
                "ub": ub,
                "missing_inc": miss_inc,
                "missing_cand": miss_c,
                "entering": n_ent,
                "seconds": time.perf_counter() - t0,
            }
        )
        if rnd % 5 == 0 or not (miss_inc or miss_c or n_ent) or ub - lp <= 0.25:
            print(
                f"    round {rnd}: |W|={len(cand)} inc edges {len(e_inc[0]):,} LP={lp - V0:.1f} UB={ub - V0:.1f} best={L - V0} missing {miss_inc}+{miss_c} entering {n_ent} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        if (miss_inc or miss_c or n_ent) and ub - lp > 0.25:
            continue
        # prices optimal for the LP of the whole instance (nothing is missing, or the bound of the whole instance already equals the LP value
        # of the working sets, which no price can beat): decide the optimum on the candidates and edges that can still matter
        for attempt in range(3):
            if ub < L + 1 - tol:
                return {
                    "closed": True,
                    "value": L,
                    "selected": best,
                    "bound": ub,
                    "rounds": rnd + 1,
                    "log": log[-3:],
                    "seconds": time.perf_counter() - t0,
                    "mip": None,
                }
            if lp_only:
                return {
                    "closed": False,
                    "value": L,
                    "selected": best,
                    "bound": ub,
                    "rounds": rnd + 1,
                    "log": log[-3:],
                    "seconds": time.perf_counter() - t0,
                    "mip": "not attempted after an open smaller budget",
                }
            slack = ub - L - 1 + tol
            rc = np.minimum(U - mu, 0.0)
            must = np.flatnonzero(P.act & (rc >= -slack))
            if len(must) > 3000 and attempt == 0:  # first a MIP on the working sets: its selection usually closes most of the gap
                target, t_inc, t_c = cand, e_inc, (ce_w, ce_q)
                exact_sets = False
            else:
                thr = t - slack
                iw, iq, n = edges_above(pi, P.A0T, thr, edge_cap)
                mq, mv = np.empty((len(must), K), np.int64), np.empty((len(must), K), np.float64)
                mT, mt = np.zeros(len(must)), np.zeros(len(must))
                row_top(pi, np.ascontiguousarray(P.AC[must]).astype(np.int64), P.bC[must], K, mq, mv, mT, mt)
                cw, cq2, n2 = edges_above(pi, np.ascontiguousarray(P.AC[must]).astype(np.int64), mt - (slack + rc[must]), edge_cap)
                if n < 0 or n2 < 0:
                    return {
                        "closed": False,
                        "value": L,
                        "selected": best,
                        "bound": ub,
                        "rounds": rnd + 1,
                        "log": log[-3:],
                        "seconds": time.perf_counter() - t0,
                        "mip": "too many edges",
                    }
                target, t_inc, t_c = must, (iw[:n], iq[:n]), (cw[:n2], cq2[:n2])
                exact_sets = True
            print(
                f"    gap {ub - L:.1f}: MIP on {len(target)} candidates, {len(t_inc[0]):,} incumbent edges, {len(t_c[0]):,} candidate edges ({'all that can matter' if exact_sets else 'working sets'})",
                flush=True,
            )
            left = limit - (time.perf_counter() - t0)
            L_before = L
            val, _, _, sel, status = P.solve(
                target,
                t_inc,
                t_c,
                True,
                max(60.0, left) if exact_sets else max(60.0, HEUR * left),
                cutoff=(L + 1 - tol) if (exact_sets and not plain) else None,
            )
            if sel:
                v = P.value(sel)
                if v > L:
                    L, best = v, sel
            print(f"    MIP: value {L - V0}, status {status} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if not exact_sets:  # the working sets only serve to find a better selection
                continue
            if status == 0 and L == L_before and not plain:
                # the row "value >= L + 1/2" was met only within the solver's tolerances: the same MIP without the row decides
                print("    the cutoff row was met within tolerances only: solving again without it", flush=True)
                val, _, _, sel, status = P.solve(target, t_inc, t_c, True, max(60.0, limit - (time.perf_counter() - t0)))
                if sel:
                    v = P.value(sel)
                    if v > L:
                        L, best = v, sel
                print(f"    MIP without the row: value {L - V0}, status {status} ({time.perf_counter() - t0:.0f}s)", flush=True)
                if status == 0:
                    return {
                        "closed": True,
                        "value": L,
                        "selected": best,
                        "bound": ub,
                        "rounds": rnd + 1,
                        "log": log[-3:],
                        "seconds": time.perf_counter() - t0,
                        "mip": {
                            "candidates": int(len(target)),
                            "inc_edges": int(len(t_inc[0])),
                            "cand_edges": int(len(t_c[0])),
                            "plain_after_cutoff": True,
                        },
                    }
            if status == 2 and exact_sets:  # infeasible: no assignment on the candidates and edges that could beat L, so L is the optimum
                return {
                    "closed": True,
                    "value": L,
                    "selected": best,
                    "bound": ub,
                    "rounds": rnd + 1,
                    "log": log[-3:],
                    "seconds": time.perf_counter() - t0,
                    "mip": "infeasible",
                }
            if status != 0:
                return {
                    "closed": False,
                    "value": L,
                    "selected": best,
                    "bound": ub,
                    "rounds": rnd + 1,
                    "log": log[-3:],
                    "seconds": time.perf_counter() - t0,
                    "mip": f"status {status}",
                }
            if exact_sets:
                return {
                    "closed": True,
                    "value": L,
                    "selected": best,
                    "bound": ub,
                    "rounds": rnd + 1,
                    "log": log[-3:],
                    "seconds": time.perf_counter() - t0,
                    "mip": {"candidates": int(len(target)), "inc_edges": int(len(t_inc[0])), "cand_edges": int(len(t_c[0]))},
                }
        return {
            "closed": False,
            "value": L,
            "selected": best,
            "bound": ub,
            "rounds": rnd + 1,
            "log": log[-3:],
            "seconds": time.perf_counter() - t0,
            "mip": "gap too large for the tests",
        }
    return {
        "closed": False,
        "value": L,
        "selected": best,
        "bound": log[-1]["ub"],
        "rounds": len(log),
        "log": log[-5:],
        "seconds": time.perf_counter() - t0,
    }
