"""Benchmark-free validation on the 2,000 held-out validation authors.

temporal_validation   checkpoint selection during training: the first 500 validation episodes, every query against
                      every history, val_score = 0.5 (PC acc + RC acc) + 0.4 (MRR + R@1)
validation_suite      the model-selection composite: the mean of four MRRs over the 2,000 validation episodes (all
                      2,000 candidates, the 50 most topic-similar non-co-author candidates, and both again with each
                      history cut to its 5 and 15 newest papers)
long_history          the same 2,000 authors with their full earlier records (up to 256 papers), scored with the
                      benchmark support budget. The selection rule reads mrr_long_all_gt48, the MRR over the authors
                      with more than 48 earlier papers.
All three score with FP32 prototypes (head.proto_fp32 = True).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import L_MAX_SCORE, L_MAX_TRAIN, L_REC_SCORE, L_REC_TRAIN
from .support import episode_supports


@torch.no_grad()
def temporal_validation(head, vecs_gpu, val_q_idx, val_h_idx, val_m, val_s_idx, val_sm) -> Dict[str, float]:
    head.eval()
    vq = vecs_gpu[val_q_idx].float()
    vh = vecs_gpu[val_h_idx].float()
    vhs = vecs_gpu[val_s_idx].float()
    N = vq.shape[0]
    sim_all = head.score_all_pairs(vq, vh, val_m, vhs, val_sm)["scores"]
    pos = sim_all.diag().unsqueeze(1)
    eye = torch.eye(N, dtype=torch.bool, device=sim_all.device)
    pc_acc = 100.0 * float(((pos > sim_all) & (~eye)).float().sum().item() / (~eye).float().sum().item())
    rc_acc = 100.0 * float(((pos > sim_all.T) & (~eye)).float().sum().item() / (~eye).float().sum().item())
    ranks = (sim_all >= pos).sum(dim=-1).float()
    r1 = 100.0 * float((ranks == 1).float().mean().item())
    r5 = 100.0 * float((ranks <= 5).float().mean().item())
    mrr = 100.0 * float((1.0 / ranks).mean().item())
    val_score = 0.50 * (pc_acc + rc_acc) + 0.40 * (mrr + r1)
    return {"val_pc_acc": round(pc_acc, 2), "val_rc_acc": round(rc_acc, 2), "val_r1": round(r1, 2), "val_r5": round(r5, 2),
            "val_mrr": round(mrr, 2), "val_score": round(val_score, 3)}


def csr_sets(indptr, idx):
    return [set(idx[indptr[i]:indptr[i + 1]].tolist()) for i in range(len(indptr) - 1)]


@torch.no_grad()
def _hard_pools(vecs_gpu, vh, vm, vco_indptr, vco_idx, device, k=50):
    """For each validation author, the k other authors whose history centroids are closest, co-authors excluded."""
    H = vecs_gpu[torch.from_numpy(vh).long().to(device)].float() * torch.from_numpy(vm).to(device)[:, :, None]
    cent = F.normalize(H.sum(1), dim=-1)
    C = cent @ cent.T
    C.fill_diagonal_(-9.0)
    for i, s in enumerate(csr_sets(vco_indptr, vco_idx)):
        if s:
            C[i, torch.tensor(sorted(s), device=device)] = -9.0
    return C.topk(k, dim=1).indices


@torch.no_grad()
def validation_suite(head, vecs_gpu, pack, device) -> Dict[str, object]:
    vq, vh, vm = pack["val_q_idx"], pack["val_h_idx"], pack["val_m"]
    N = len(vq)
    vecs_np = pack["vecs_train"]
    Q = vecs_gpu[torch.from_numpy(vq).long().to(device)].float()

    def S_matrix(h_idx, h_m):
        s_idx, s_m = episode_supports(vecs_np, h_idx, h_m, L_MAX_TRAIN, L_REC_TRAIN)
        H = vecs_gpu[torch.from_numpy(h_idx).long().to(device)].float()
        HM = torch.from_numpy(h_m).to(device)
        HS = vecs_gpu[torch.from_numpy(s_idx).long().to(device)].float()
        HSM = torch.from_numpy(s_m).to(device)
        S = torch.zeros(N, N, device=device)
        for s in range(0, N, 100):
            S[s:s + 100] = head.score_all_pairs(Q[s:s + 100], H, HM, HS, HSM)["scores"].float()
        return S

    rr = {}

    def mrr_r1(S, pool=None, key=None):
        pos = S.diag()
        rank = (S >= pos[:, None]).sum(1).float() if pool is None else 1.0 + (S.gather(1, pool) >= pos[:, None]).sum(1).float()
        if key is not None:
            rr[key] = (1.0 / rank).cpu().numpy().astype(np.float32)
        return round(100.0 * float((1.0 / rank).mean()), 3), round(100.0 * float((rank == 1).float().mean()), 3)

    pool = _hard_pools(vecs_gpu, vh, vm, pack["vco_indptr"], pack["vco_idx"], device)
    res = {}
    S = S_matrix(vh, vm)
    res["mrr_all2000"], res["r1_all2000"] = mrr_r1(S, key="all2000")
    res["mrr_hard50"], res["r1_hard50"] = mrr_r1(S, pool, key="hard50")
    for k in (5, 15):
        Sk = S_matrix(vh[:, :k].copy(), vm[:, :k].copy())
        res[f"mrr_short{k}"], res[f"r1_short{k}"] = mrr_r1(Sk, key=f"short{k}")
        res[f"mrr_short{k}_hard50"], _ = mrr_r1(Sk, pool, key=f"short{k}_hard50")
    res["composite"] = round((res["mrr_all2000"] + res["mrr_hard50"] + res["mrr_short5"] + res["mrr_short15"]) / 4.0, 3)
    res["_rr"] = rr  # per-query reciprocal ranks
    return res


@torch.no_grad()
def long_history(head, lv, device) -> Dict[str, float]:
    """lv: the arrays of data/train/<encoder>/longval.npz (V, q, h, m, val_vco_indptr, val_vco_idx)."""
    V = torch.from_numpy(lv["V"]).to(device)
    q, h, m = lv["q"], lv["h"], lv["m"]
    N = len(q)
    Q = V[torch.from_numpy(q).long().to(device)].float()
    s_idx, s_m = episode_supports(lv["V"], h, m, L_MAX_SCORE, L_REC_SCORE)
    S = torch.zeros(N, N, device=device)
    for a0 in range(0, N, 400):
        a1 = min(N, a0 + 400)
        H = V[torch.from_numpy(h[a0:a1]).long().to(device)].float()
        HM = torch.from_numpy(m[a0:a1]).to(device)
        HS = V[torch.from_numpy(s_idx[a0:a1]).long().to(device)].float()
        HSM = torch.from_numpy(s_m[a0:a1]).to(device)
        for s in range(0, N, 200):
            S[s:s + 200, a0:a1] = head.score_all_pairs(Q[s:s + 200], H, HM, HS, HSM)["scores"].float()
    pool = _hard_pools(V, h, m, lv["val_vco_indptr"], lv["val_vco_idx"], device)
    pos = S.diag()
    r_all = (S >= pos[:, None]).sum(1).float()
    r_hard = 1.0 + (S.gather(1, pool) >= pos[:, None]).sum(1).float()
    long_ = torch.from_numpy(m.sum(1)).to(device) > 48
    res = {}
    for nm, r in (("all", r_all), ("hard50", r_hard)):
        res[f"mrr_long_{nm}"] = round(100 * float((1 / r).mean()), 3)
        res[f"mrr_long_{nm}_gt48"] = round(100 * float((1 / r[long_]).mean()), 3)
        res[f"mrr_long_{nm}_le48"] = round(100 * float((1 / r[~long_]).mean()), 3)
    res["composite_long"] = round((res["mrr_long_all"] + res["mrr_long_hard50"]) / 2, 3)
    res["n_gt48"] = int(long_.sum())
    return res
