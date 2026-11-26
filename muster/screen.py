"""Negative-only screen of in-batch negatives (Section 4.1 of the paper).

For a submission q and a reviewer r, represented by the support set of r's training episode, three sources of evidence:
  Z  authorship: q is in r's record
  T  text: the largest frozen cosine between q and the support papers reaches t_T
  G  citations: the largest Jaccard similarity between the reference list of q and those of the support papers reaches
     t_G; G = 0 only if every list is usable, otherwise G is missing (2). A query without a usable list makes G missing.
A two-class latent-class model per history-length group h (tertiles) gives the posterior p(E = 1 | h, Z, T, G) that r is
qualified for q. The model is fitted by EM before training and cross-fitted over two folds of the papers, so each
posterior comes from the fold that did not contain its paper. Authors are never screened. A non-author stays a negative
only if its posterior is at most delta, in both the paper-centric and the reviewer-centric comparisons.
"""
from __future__ import annotations

import json
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch


class ReferenceIndex:
    """Reference lists of the training papers as a sparse binary matrix (rows = corpus rows, columns = cited works)."""

    def __init__(self, path) -> None:
        z = np.load(path)
        n = len(z["indptr"]) - 1
        self.R = sp.csr_matrix((np.ones(len(z["indices"]), dtype=np.float32), z["indices"], z["indptr"]),
                               shape=(n, int(z["n_vocab"])))
        self.size = np.diff(z["indptr"]).astype(np.float32)
        self.has = z["has_refs"].astype(bool)

    def g_values(self, q_rows, s_rows, s_mask):
        """q_rows (B,), support rows s_rows (N, L) with mask s_mask (N, L). Returns v_G (B, N) with NaN where undefined,
        the usable support papers per reviewer (N,), the support papers per reviewer (N,) and whether each query has a
        usable list (B,)."""
        B, (N, L) = len(q_rows), s_rows.shape
        flat = s_rows.reshape(-1)
        inter = (self.R[q_rows] @ self.R[flat].T).toarray().reshape(B, N, L)
        sq, ss = self.size[q_rows][:, None, None], self.size[flat].reshape(1, N, L)
        union = sq + ss - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        usable = s_mask & self.has[s_rows]
        jac = np.where(usable[None, :, :], jac, -1.0)
        v = jac.max(axis=2)
        q_has = self.has[q_rows]
        v = np.where((v < 0) | ~q_has[:, None], np.nan, v)
        return v.astype(np.float32), usable.sum(1), s_mask.sum(1), q_has


@torch.no_grad()
def t_values(vq: torch.Tensor, vs: torch.Tensor, s_mask: torch.Tensor) -> torch.Tensor:
    """Frozen query vectors vq (B, d), support vectors vs (N, L, d) with mask (N, L) -> v_T (B, N)."""
    return torch.einsum("bd,nld->bnl", vq, vs).masked_fill(~s_mask.unsqueeze(0), -9.0).max(dim=-1).values


def categories(v_t, v_g, q_has, n_use, n_supp, t_t, t_g):
    """T in {0, 1} and G in {0, 1, 2 = missing}. q_has, n_use and n_supp broadcast to the shape of v_g."""
    T = (v_t >= t_t).astype(np.int8)
    defined = ~np.isnan(v_g)
    vg = np.nan_to_num(v_g, nan=-1.0)
    complete = np.broadcast_to((n_use == n_supp) & q_has, v_g.shape)
    G = np.full(v_g.shape, 2, dtype=np.int8)
    G[defined & (vg >= t_g)] = 1
    G[defined & (vg < t_g) & complete] = 0
    return T, G


def author_row_sets(seq_indptr: np.ndarray, seq_idx: np.ndarray) -> List[Set[int]]:
    """The corpus rows of each training author's record."""
    return [set(seq_idx[seq_indptr[a]:seq_indptr[a + 1]].tolist()) for a in range(len(seq_indptr) - 1)]


def fold_of(q_rows) -> np.ndarray:
    """Cross-fitting fold (0 or 1) of each paper row."""
    return ((np.asarray(q_rows, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32) % np.uint64(2)).astype(np.int8)


class LatentClassModel:
    """Posterior table[fold of the query][h][Z][T][G] = p(E = 1), where the table for fold k was fitted on the other fold."""

    def __init__(self, path) -> None:
        m = json.loads(open(path).read())
        self.t_t, self.t_g, self.cuts = m["t_T"], m["t_G"], np.asarray(m["h_cuts"])
        self.table = np.asarray(m["posterior_for_fold"], dtype=np.float32)  # (2, 3, 2, 2, 3)
        self.meta = m

    def h_groups(self, n_hist: np.ndarray) -> np.ndarray:
        return np.digitize(n_hist, self.cuts, right=True).astype(np.int8)  # n <= c1 -> 0, c1 < n <= c2 -> 1, else 2

    def posterior(self, fold_q, h_c, Z, T, G):
        return self.table[fold_q[:, None], h_c[None, :], Z, T, G]


class NegativeScreen:
    """Per-batch masks of the kept negatives for the training loop."""

    def __init__(self, model_path, refs_path, delta: float, seq_indptr, seq_idx, tq, tm, ta, s_idx, s_m,
                 device: torch.device) -> None:
        self.model, self.refs, self.delta = LatentClassModel(model_path), ReferenceIndex(refs_path), float(delta)
        self.authors = author_row_sets(seq_indptr, seq_idx)
        self.tq, self.ta = np.asarray(tq), np.asarray(ta)
        self.h = self.model.h_groups(np.asarray(tm).sum(1))
        self.s_idx, self.s_m = np.asarray(s_idx), np.asarray(s_m).astype(bool)
        self.dev = device
        self.acc = {"steps": 0, "pc_kept": 0.0, "rc_kept": 0.0, "anchors_with_pc": 0.0, "anchor_p_mean": 0.0}

    def masks(self, idx_np: np.ndarray, bq: torch.Tensor, bhs: torch.Tensor, bms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Episodes idx_np of the batch, their frozen query vectors bq (B, d) and support vectors bhs (B, L, d) with mask
        bms. Returns the paper-centric mask pc[i, j] (episode j's author is a negative for query i) and the
        reviewer-centric mask rc[i, j] (query j is a negative for episode i's author), both (B, B)."""
        q = self.tq[idx_np]
        v_t = t_values(bq.float(), bhs.float(), bms).cpu().numpy()
        v_g, n_use, n_sup, q_has = self.refs.g_values(q, self.s_idx[idx_np], self.s_m[idx_np])
        T, G = categories(v_t, v_g, q_has[:, None], n_use[None, :], n_sup[None, :], self.model.t_t, self.model.t_g)
        Z = np.array([[int(qq) in self.authors[int(a)] for a in self.ta[idx_np]] for qq in q], dtype=np.int8)
        P = self.model.posterior(fold_of(q), self.h[idx_np], Z, T, G)
        a = np.diag(P)
        off = ~np.eye(len(q), dtype=bool)
        pc = off & (Z == 0) & (P <= self.delta)
        rc = off & (Z.T == 0) & (P.T <= self.delta)
        k = self.acc
        k["steps"] += 1
        k["pc_kept"] += pc.sum() / max(1, (off & (Z == 0)).sum())
        k["rc_kept"] += rc.sum() / max(1, (off & (Z.T == 0)).sum())
        k["anchors_with_pc"] += (pc.sum(1) > 0).mean()
        k["anchor_p_mean"] += a.mean()
        return torch.from_numpy(pc).to(self.dev), torch.from_numpy(rc).to(self.dev)

    def summary(self) -> Dict[str, float]:
        n = max(1, self.acc["steps"])
        return {"delta": self.delta, "steps": self.acc["steps"],
                **{k: round(float(v) / n, 4) for k, v in self.acc.items() if k != "steps"}}
