"""Confidence-gated self-paced curriculum (Section 4.2 of the paper).

The curriculum state s_t mixes training progress with the model's competence. A degree-6 Bernstein schedule in s_t sets
the proportions of easy, semi-hard and hard negatives, the margin mu and the temperature tau_l of the ranking loss. Each
negative's weight is its band proportion shared over the band's negatives, times a confidence gate that is strictest in
the hard band.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

DEGREE = 6
GAMMA_TIME = 0.55  # weight of training progress in the curriculum state


def bernstein_schedule(step: int, total_steps: int, p_competence: float, degree: int = DEGREE,
                       gamma_time: float = GAMMA_TIME) -> Dict[str, float]:
    s_time = min(max(float(step) / float(max(total_steps, 1)), 0.0), 1.0)
    p_norm = min(max((p_competence - 0.10) / 0.65, 0.0), 1.0)
    s_eff = min(max(gamma_time * s_time + (1.0 - gamma_time) * p_norm, 0.0), 1.0)
    B = degree
    basis = np.array([math.comb(B, k) * (s_eff ** k) * ((1.0 - s_eff) ** (B - k)) for k in range(B + 1)], dtype=np.float64)
    ctrl_easy = np.linspace(0.72, 0.14, B + 1)
    ctrl_semi = np.array([0.22, 0.28, 0.34, 0.38, 0.38, 0.36, 0.34], dtype=np.float64)[: B + 1]
    ctrl_hard = np.maximum(0.0, 1.0 - ctrl_easy - ctrl_semi)
    ctrl_margin = np.linspace(0.055, 0.100, B + 1)
    ctrl_temp = np.linspace(0.095, 0.075, B + 1)
    p_e = float(np.dot(basis, ctrl_easy))
    p_s = float(np.dot(basis, ctrl_semi))
    p_h = float(np.dot(basis, ctrl_hard))
    tot = max(p_e + p_s + p_h, 1e-8)
    return {"s_eff": float(s_eff), "p_easy": p_e / tot, "p_semi": p_s / tot, "p_hard": p_h / tot,
            "margin": float(np.dot(basis, ctrl_margin)), "tau_l": float(np.dot(basis, ctrl_temp))}


def pair_weights(deltas: torch.Tensor, sim_teach: torch.Tensor, valid_neg: torch.Tensor,
                 sched: Mapping[str, float]) -> Tuple[torch.Tensor, float]:
    """Weights of the kept negatives (B, N) and the batch competence: the share of the 16 most teacher-similar kept
    negatives of each row whose score gap Delta reaches the margin. deltas = score of the positive minus score of the
    negative, sim_teach = teacher score s0 of each comparison."""
    valid_f = valid_neg.float()
    margin = float(sched["margin"])
    tau_l = float(sched["tau_l"])
    with torch.no_grad():
        s_min = sim_teach.masked_fill(~valid_neg, 1e4).min(dim=-1, keepdim=True).values
        s_max = sim_teach.masked_fill(~valid_neg, -1e4).max(dim=-1, keepdim=True).values
        s_norm = ((sim_teach - s_min) / (s_max - s_min).clamp(min=1e-4)).clamp(0.0, 1.0)
        conf = (1.0 - 0.60 * s_norm).clamp(0.20, 0.98)

        k_eval = min(16, sim_teach.shape[1])
        top_idx = sim_teach.masked_fill(~valid_neg, -1e4).topk(k_eval, dim=-1).indices
        top_deltas = torch.gather(deltas, 1, top_idx)
        top_valid = torch.gather(valid_neg, 1, top_idx)
        p_batch = float(((top_deltas >= margin) & top_valid).float().sum().item()
                        / top_valid.float().sum().clamp(min=1.0).item())

        rho_e, rho_s, rho_h = float(sched["p_easy"]), float(sched["p_semi"]), float(sched["p_hard"])
        tau_c = torch.sigmoid((margin - deltas) / tau_l)
        gate_c = torch.sigmoid((conf - 0.75 * tau_c) / 0.075)
        is_easy = (deltas >= margin) & valid_neg
        is_semi = (deltas > 0.0) & (deltas < margin) & valid_neg
        is_hard = (deltas <= 0.0) & valid_neg
        n_e = is_easy.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        n_s = is_semi.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        n_h = is_hard.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        band_w = rho_e * (is_easy.float() / n_e) + rho_s * (is_semi.float() / n_s) + rho_h * (is_hard.float() / n_h)
        w = band_w * gate_c * valid_f
    return w.detach(), p_batch
