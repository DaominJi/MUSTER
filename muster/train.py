"""Training of one MUSTER head on frozen embeddings (Section 4 of the paper).

Each training episode e holds a paper q_e of an author r_e and the author's earlier papers H_e (newest first). A step draws
a minibatch of episodes. For each query, the other episodes' authors are candidate negatives, kept only if they pass the
negative-only screen. The loss is the paper-centric ranking loss plus the reviewer-centric one, weighted by the
confidence-gated curriculum, plus self-distillation from the frozen teacher score and two prototype regularizers
(coverage of the support papers and diversity of the prototypes). An EMA of the weights is validated every val_every
steps on held-out authors, and the best EMA checkpoint is returned.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import GAMMA_TOP1, GAMMA_TOP3, L_MAX_TRAIN, L_REC_TRAIN, TrainConfig
from .curriculum import bernstein_schedule, pair_weights
from .model import MusterHead
from .screen import NegativeScreen
from .support import episode_supports
from .validate import temporal_validation


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_scores(bq: torch.Tensor, bh: torch.Tensor, bm: torch.Tensor) -> torch.Tensor:
    """s0 on the frozen embeddings: GAMMA_TOP1 * top-1 + GAMMA_TOP3 * mean of the top-3 cosines (B, N)."""
    raw_sim = torch.einsum("bd,rld->brl", bq, bh).masked_fill(~bm.unsqueeze(0), -1e4)
    t_top1 = raw_sim.max(dim=-1).values
    t_top3 = raw_sim.topk(min(3, raw_sim.shape[-1]), dim=-1).values.mean(dim=-1)
    return GAMMA_TOP1 * t_top1 + GAMMA_TOP3 * t_top3


def train_head(pack: Dict[str, np.ndarray], screen_model: str, screen_refs: str, cfg: TrainConfig, device: torch.device,
               log: Callable[[str], None] = print) -> Tuple[MusterHead, Dict[str, Any]]:
    """Train one head from pack arrays (data/train/episodes.npz and vectors) and return the best EMA head and summary."""
    t0 = time.time()
    T_STEPS, B = int(cfg.steps), int(cfg.batch)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    vecs = pack["vecs_train"]
    vecs_gpu = torch.from_numpy(vecs).to(device)
    tq, th, tm, ta = pack["train_q_idx"], pack["train_h_idx"], pack["train_m"], pack["ep_author"]
    E = len(tq)
    n_val = int(cfg.val_episodes)
    val_q, val_h, val_m = pack["val_q_idx"][:n_val], pack["val_h_idx"][:n_val], pack["val_m"][:n_val]
    train_s_idx, train_sm = episode_supports(vecs, th, tm, L_MAX_TRAIN, L_REC_TRAIN)
    val_s_idx, val_sm = episode_supports(vecs, val_h, val_m, L_MAX_TRAIN, L_REC_TRAIN)
    log(f"episodes {E:,} | support sets ready in {time.time() - t0:.0f}s")
    screen = NegativeScreen(screen_model, screen_refs, cfg.screen_delta, pack["seq_indptr"], pack["seq_idx"], tq, tm, ta,
                            train_s_idx, train_sm, device)

    head = MusterHead(cfg.model).to(device)
    head.set_corpus_mean(pack["mu_train"])
    head.proto_fp32 = False
    tq_i = torch.from_numpy(tq).long().to(device)
    th_i = torch.from_numpy(th).long().to(device)
    tm_b = torch.from_numpy(tm).bool().to(device)
    ts_i = torch.from_numpy(train_s_idx).long().to(device)
    tsm_b = torch.from_numpy(train_sm).bool().to(device)
    vq_i, vh_i = torch.from_numpy(val_q).long().to(device), torch.from_numpy(val_h).long().to(device)
    vm_b, vs_i = torch.from_numpy(val_m).bool().to(device), torch.from_numpy(val_s_idx).long().to(device)
    vsm_b = torch.from_numpy(val_sm).bool().to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = max(60, int(cfg.warmup_frac * T_STEPS))

    def lr_lambda(s):
        if s < warmup:
            return float(s + 1) / float(warmup)
        prog = float(s - warmup) / float(max(1, T_STEPS - warmup))
        return cfg.lr_floor + (1.0 - cfg.lr_floor) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    ema_head = copy.deepcopy(head).to(device)
    ema_head.proto_fp32 = True  # checkpoint validation in FP32
    init_val = temporal_validation(ema_head, vecs_gpu, vq_i, vh_i, vm_b, vs_i, vsm_b)
    best_val, best_state, best_step = init_val["val_score"], copy.deepcopy(ema_head.state_dict()), 0
    p_competence = float(cfg.p0)
    traj = []

    perm = rng.permutation(E)
    ptr = 0
    for step in range(1, T_STEPS + 1):
        if ptr + B > len(perm):
            perm = rng.permutation(E)
            ptr = 0
        idx_np = np.asarray(perm[ptr:ptr + B], dtype=np.int64)
        ptr += B
        nb = len(idx_np)
        idx = torch.from_numpy(idx_np).long().to(device)
        bq = vecs_gpu[tq_i[idx]].float()
        bh, bm = vecs_gpu[th_i[idx]].float(), tm_b[idx]
        bhs, bms = vecs_gpu[ts_i[idx]].float(), tsm_b[idx]
        with torch.no_grad():
            sim_teach = teacher_scores(bq, bh, bm)
        keep_pc, keep_rc = screen.masks(idx_np, bq, bhs, bms)
        b_sched = bernstein_schedule(step, T_STEPS, p_competence)
        margin, tau_l = b_sched["margin"], b_sched["tau_l"]
        head.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp_bf16 and device.type == "cuda"):
            out = head.score_all_pairs(bq, bh, bm, bhs, bms)
            scores = out["scores"]
            pos = torch.diag(scores[:, :nb]).unsqueeze(1)
            mask_neg = torch.ones(nb, nb, dtype=torch.bool, device=device)
            mask_neg[:, :nb] = ~torch.eye(nb, dtype=torch.bool, device=device)
            valid_neg = mask_neg & keep_pc
            # paper-centric: the author of q_e against the other episodes' authors
            deltas_pc = pos - scores
            w_pc, p_b = pair_weights(deltas_pc, sim_teach, valid_neg, b_sched)
            p_competence = 0.98 * p_competence + 0.02 * p_b
            loss_pc = (F.softplus((margin - deltas_pc) / tau_l) * w_pc).sum() / w_pc.sum().clamp(min=1e-6)
            # reviewer-centric: for the history of e, its own paper against the other episodes' papers
            sq = scores[:, :nb]
            deltas_rc = pos - sq.T
            w_rc, _ = pair_weights(deltas_rc, sim_teach[:, :nb].T, keep_rc, b_sched)
            loss_rc = (F.softplus((margin - deltas_rc) / tau_l) * w_rc).sum() / w_rc.sum().clamp(min=1e-6)
            # prototype coverage of the history papers and diversity among the prototypes
            protos, pmask, h_hat = out["protos"][:nb], out["pmask"][:nb], out["h_hat"][:nb]
            bm = bm[:nb]
            sim_hp = torch.einsum("bld,bmd->blm", h_hat, protos).masked_fill(~pmask.unsqueeze(1), -1e4)
            loss_cov = ((1.0 - sim_hp.max(dim=-1).values) * bm.float()).sum() / bm.float().sum().clamp(min=1.0)
            gram = torch.einsum("bmd,bnd->bmn", protos, protos)
            eye_m = torch.eye(protos.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
            p_valid = pmask.unsqueeze(2) & pmask.unsqueeze(1) & (~eye_m)
            loss_div = ((F.relu(gram - cfg.div_margin) ** 2) * p_valid.float()).sum() / p_valid.float().sum().clamp(min=1.0)
            # self-distillation from the teacher's softmax over the batch's histories
            p_t_dist = F.softmax(sim_teach / cfg.tau_kd, dim=-1)
            loss_kd = F.kl_div(F.log_softmax(scores / cfg.tau_kd, dim=-1), p_t_dist, reduction="batchmean") * (cfg.tau_kd ** 2)
            loss = loss_pc + cfg.lam_rc * loss_rc
            loss = loss + cfg.lam_kd * loss_kd
            loss = loss + cfg.lam_cov * loss_cov
            loss = loss + cfg.lam_div * loss_div
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=cfg.clip)
        opt.step()
        sched_lr.step()
        with torch.no_grad():
            for p_ema, p_cur in zip(ema_head.parameters(), head.parameters()):
                p_ema.data.mul_(cfg.ema).add_(p_cur.data, alpha=1.0 - cfg.ema)
        if step == 1 or step % cfg.val_every == 0 or step == T_STEPS:
            vm = temporal_validation(ema_head, vecs_gpu, vq_i, vh_i, vm_b, vs_i, vsm_b)
            if step >= cfg.val_every and vm["val_score"] >= best_val:
                best_val, best_state, best_step = vm["val_score"], copy.deepcopy(ema_head.state_dict()), step
            traj.append({"step": step, "loss": round(float(loss.item()), 4), "l_kd": round(float(loss_kd.item()), 5),
                         "l_cov": round(float(loss_cov.item()), 5), "l_div": round(float(loss_div.item()), 5), **vm,
                         "elapsed_sec": round(time.time() - t0, 1)})
            log(f"step {step}/{T_STEPS} loss {loss.item():.4f} val {vm['val_score']:.3f} (MRR {vm['val_mrr']:.2f}) "
                f"{time.time() - t0:.0f}s")
    ema_head.load_state_dict(best_state)
    ema_head.eval()
    info = {"episodes": E, "steps": T_STEPS, "best_step": best_step, "best_val_score_500": round(float(best_val), 3),
            "train_sec": round(time.time() - t0, 1), "trajectory": traj, "screen": screen.summary()}
    return ema_head, info
