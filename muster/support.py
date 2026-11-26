"""Support sets: the recent and representative papers that the core branch summarizes into prototypes.

A history is a list of paper rows ordered from newest to oldest. A history of at most l_max papers is kept whole.
Otherwise the support set keeps the l_rec newest papers and splits the older papers into l_max - l_rec consecutive
chronological segments. From each segment it keeps the paper with the highest mean cosine to the papers of its segment,
minus LAMBDA_RED times its largest cosine to the newest papers (Eq. support_select of the paper).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .config import LAMBDA_RED


def select_support_rows(rows: Sequence[int], X: np.ndarray, l_max: int, l_rec: int,
                        lambda_red: float = LAMBDA_RED) -> List[int]:
    """Support set of one history. `rows` index the unit vectors X and run from newest to oldest."""
    rows_list = list(rows)
    if len(rows_list) <= l_max:
        return rows_list
    rec = rows_list[:l_rec]
    older = rows_list[l_rec:]
    H_rec = X[rec]
    selected_older: List[int] = []
    for s_idx in np.array_split(np.arange(len(older)), l_max - l_rec):
        if len(s_idx) == 0:
            continue
        if len(s_idx) == 1:
            selected_older.append(older[int(s_idx[0])])
            continue
        cand_rows = [older[int(i)] for i in s_idx]
        H_b = X[cand_rows]
        rep_scores = (H_b @ H_b.T).mean(axis=1)  # typicality within the segment
        red_scores = (H_b @ H_rec.T).max(axis=1)  # redundancy to the newest papers
        selected_older.append(cand_rows[int(np.argmax(rep_scores - lambda_red * red_scores))])
    return rec + selected_older


def episode_supports(vecs: np.ndarray, h_idx: np.ndarray, h_mask: np.ndarray, l_max: int,
                     l_rec: int) -> Tuple[np.ndarray, np.ndarray]:
    """Support sets of many histories at once. h_idx (E, L) holds corpus rows, newest first, valid where h_mask (E, L)
    is True (a prefix of each row). Selection reads the frozen vectors. Returns rows (E, l_max) and mask (E, l_max)."""
    E = h_idx.shape[0]
    supp_idx = np.zeros((E, l_max), dtype=np.int32)
    supp_mask = np.zeros((E, l_max), dtype=bool)
    for e_i in range(E):
        n_i = int(h_mask[e_i].sum())
        if n_i == 0:
            continue
        if n_i <= l_max:
            supp_idx[e_i, :n_i] = h_idx[e_i, :n_i]
            supp_mask[e_i, :n_i] = True
        else:
            rows_e = h_idx[e_i, :n_i]
            sel_local = select_support_rows(list(range(n_i)), X=vecs[rows_e].astype(np.float32), l_max=l_max, l_rec=l_rec)
            supp_idx[e_i, :len(sel_local)] = rows_e[sel_local]
            supp_mask[e_i, :len(sel_local)] = True
    return supp_idx, supp_mask
