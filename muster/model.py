"""The MUSTER head: dual-branch expertise matching on a frozen paper encoder (Section 3 of the paper).

The head maps a submission q and a reviewer history (unit vectors from one frozen encoder) to an expertise score:
  adapter      e -> e_hat: corpus-mean centering with a learned strength and a residual MLP that starts at the identity
  core branch  a support set of the history -> pre-norm set attention blocks -> M prototypes seeded at evenly spaced
               support positions, refined by cross- and self-attention -> S_core, a mass-weighted smooth maximum
  niche branch attention from the submission to every history paper, with keys taken from distinctive embeddings
               (the profile centroid partly subtracted) -> S_niche
  fusion       a gate on (alignment with the centroid, S_core - S_niche, log(1 + n_r), coherence) mixes the two scores,
               plus a per-reviewer offset on the history length and the coherence

Two scoring paths compute the same function. `score_all_pairs` scores every submission of a batch against every history
of the batch (training and validation). `score_items` scores benchmark items (submission, reviewer) one reviewer at a time.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import L_MAX_SCORE, L_REC_SCORE, ModelConfig
from .support import select_support_rows

# Upper clamps of the learned gates and strengths.
CENTER_MAX, ADAPTER_GATE_MAX, CONTEXT_GATE_MAX = 0.65, 0.35, 0.45
REFINE_GATE_MAX, QUERY_GATE_MAX, NICHE_GATE_MAX, FUSION_SCALE_MAX = 0.40, 0.25, 0.30, 0.50
LOG_LEN_NORM = math.log(1.0 + L_MAX_SCORE)  # the gate reads log(1 + n_r) / log(1 + 64)


class SetNorm(nn.Module):
    """Set normalization: mean and variance over the valid papers and all features of each set, then a per-feature
    affine map. Padded papers do not enter the statistics."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mf = mask.unsqueeze(-1).float()
        n = (mf.sum(dim=(1, 2), keepdim=True) * x.shape[-1]).clamp(min=1.0)
        mu = (x * mf).sum(dim=(1, 2), keepdim=True) / n
        var = (((x - mu) ** 2) * mf).sum(dim=(1, 2), keepdim=True) / n
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SetAttentionBlock(nn.Module):
    """Pre-norm set attention block with a gated attention output:
    x + sigmoid(W_g SetNorm(x)) * Attn(SetNorm(x)), then x + FFN(SetNorm(x)). No positional embedding, so the block is
    permutation-equivariant. Padded papers are masked as keys and zeroed in the output."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.norm1 = SetNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        self.norm2 = SetNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Linear(d_model * ffn_mult, d_model))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key_pad = ~mask
        safe_key_pad = key_pad.masked_fill(key_pad.all(dim=-1, keepdim=True), False)
        h = self.norm1(x, mask)
        a, _ = self.attn(h, h, h, key_padding_mask=safe_key_pad, need_weights=False)
        x = x + torch.sigmoid(self.gate(h)) * a
        x = x + self.ffn(self.norm2(x, mask))
        return x * mask.unsqueeze(-1).to(x.dtype)


class PrototypeRefinement(nn.Module):
    """R_theta: cross-attention from the prototypes to the contextualized papers, self-attention among the prototypes and
    a feed-forward layer, each with a residual connection and layer normalization."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Linear(d_model * ffn_mult, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, protos: torch.Tensor, pmask: torch.Tensor, h_ctx: torch.Tensor, m_supp: torch.Tensor) -> torch.Tensor:
        key_pad_h = ~m_supp
        safe_pad_h = key_pad_h.masked_fill(key_pad_h.all(dim=-1, keepdim=True), False)
        key_pad_p = ~pmask
        safe_pad_p = key_pad_p.masked_fill(key_pad_p.all(dim=-1, keepdim=True), False)
        c, _ = self.cross_attn(protos, h_ctx, h_ctx, key_padding_mask=safe_pad_h, need_weights=False)
        p1 = self.norm1(protos + c)
        s, _ = self.self_attn(p1, p1, p1, key_padding_mask=safe_pad_p, need_weights=False)
        p2 = self.norm2(p1 + s)
        out = self.norm3(p2 + self.ffn(p2))
        return out * pmask.unsqueeze(-1).to(out.dtype)


def _draw_reference_rng(d: int, n_heads: int, ffn_mult: int, n_blocks: int, n_inducing: int = 64) -> None:
    """Advance PyTorch RNG state consistently for deterministic initialization matching released checkpoints."""
    for _ in range(n_blocks):
        torch.randn(1, n_inducing, d)
        nn.MultiheadAttention(d, n_heads, batch_first=True)
        nn.MultiheadAttention(d, n_heads, batch_first=True)
        nn.Linear(d, d * ffn_mult)
        nn.Linear(d * ffn_mult, d)


_SEED_TABLES: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _strided_seeds(m_supp: torch.Tensor, M: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Seed positions round(linspace(0, L_r - 1, M_r)) with M_r = min(M, L_r) for every support set, and their mask."""
    L = m_supp.shape[1]
    key = (L, M, str(m_supp.device))
    if key not in _SEED_TABLES:
        s_tab = torch.zeros((L + 1, M), dtype=torch.long, device=m_supp.device)
        p_tab = torch.zeros((L + 1, M), dtype=torch.bool, device=m_supp.device)
        for n_i in range(1, L + 1):
            m_act = min(M, n_i)
            s_tab[n_i, :m_act] = torch.linspace(0, n_i - 1, steps=m_act, device=m_supp.device).round().long()
            p_tab[n_i, :m_act] = True
        _SEED_TABLES[key] = (s_tab, p_tab)
    s_tab, p_tab = _SEED_TABLES[key]
    lengths = m_supp.sum(dim=-1).clamp(min=0, max=L)
    return s_tab[lengths], p_tab[lengths]


class MusterHead(nn.Module):
    """One MUSTER head (44,891,732 parameters with the default ModelConfig)."""

    proto_fp32 = False  # True: prototypes in FP32 even inside BF16 autocast regions (validation and scoring)

    def __init__(self, cfg: ModelConfig = ModelConfig(), reference_rng: bool = True) -> None:
        super().__init__()
        d, nh, fm = cfg.d_emb, cfg.n_heads, cfg.ffn_mult
        self.d_emb, self.n_prototypes = d, cfg.n_prototypes
        self.tau_a, self.tau_p, self.tau_c = float(cfg.tau_a), float(cfg.tau_p), float(cfg.tau_c)
        # The modules and parameters are created in the order of the code that trained the released heads, so a seeded
        # run draws the same initial weights and the parameters appear in the same order (gradient clipping sums their
        # norms in this order).

        # Adapter (Eqs. anisotropy_center and residual_adapter)
        self.register_buffer("mu_corp", torch.zeros(1, d))
        self.center_alpha = nn.Parameter(torch.tensor(float(cfg.center_init)))
        self.adapter_ln = nn.LayerNorm(d)
        self.adapter_w1 = nn.Linear(d, cfg.adapter_hidden)
        self.adapter_w2 = nn.Linear(cfg.adapter_hidden, d)
        self.adapter_gate = nn.Parameter(torch.tensor(0.10))
        nn.init.zeros_(self.adapter_w2.weight)
        nn.init.zeros_(self.adapter_w2.bias)

        # Core branch (Eqs. prototype_vec and coarse_score)
        self.in_ln = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, d)
        self.set_blocks = nn.ModuleList()  # filled at the end of the constructor
        if reference_rng:
            _draw_reference_rng(d, nh, fm, cfg.n_blocks)
        self.out_ln = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.context_gate = nn.Parameter(torch.tensor(0.18))  # gamma_u
        self.refine = PrototypeRefinement(d, nh, fm)
        self.refine_gate = nn.Parameter(torch.tensor(0.10))  # gamma_p
        self.query_mlp = nn.Sequential(nn.Linear(d, d * 2, bias=False), nn.Linear(d * 2, d, bias=False))
        nn.init.zeros_(self.query_mlp[-1].weight)
        self.query_gate = nn.Parameter(torch.tensor(0.05))  # gamma_q

        # Niche branch (Eqs. distinctive and cross_attn): W_X(v) = norm(v + gamma_n A_X v), A_X of rank niche_rank
        self.niche_q = nn.Sequential(nn.Linear(d, cfg.niche_rank, bias=False), nn.Linear(cfg.niche_rank, d, bias=False))
        self.niche_k = nn.Sequential(nn.Linear(d, cfg.niche_rank, bias=False), nn.Linear(cfg.niche_rank, d, bias=False))
        self.niche_v = nn.Sequential(nn.Linear(d, cfg.niche_rank, bias=False), nn.Linear(cfg.niche_rank, d, bias=False))
        nn.init.zeros_(self.niche_q[-1].weight)
        nn.init.zeros_(self.niche_k[-1].weight)
        nn.init.zeros_(self.niche_v[-1].weight)
        self.niche_gate = nn.Parameter(torch.tensor(0.10))  # gamma_n
        beta = float(cfg.beta_init)
        self.beta_logit = nn.Parameter(torch.tensor(math.log(beta / max(0.5 - beta, 1e-4))))  # beta = sigmoid(.) / 2

        # Gated fusion (Eq. final_score). core_logit is the logit of the core weight 1 - omega before the gate's term.
        self.core_logit = nn.Parameter(torch.tensor(math.log(cfg.omega_init / (1.0 - cfg.omega_init))))
        self.fusion_ln = nn.LayerNorm(4)
        self.fusion_mlp = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)
        self.fusion_scale = nn.Parameter(torch.tensor(0.25))
        self.offset_len = nn.Parameter(torch.tensor(-0.004))  # w_len
        self.offset_coh = nn.Parameter(torch.tensor(-0.012))  # w_coh

        # The set blocks draw their initial weights from a copy of the random state, as in the reference code.
        with torch.random.fork_rng(devices=[]):
            self.set_blocks.extend(SetAttentionBlock(d, nh, fm) for _ in range(cfg.n_blocks))

    # ------------------------------------------------------------------------------------------------------------------
    def set_corpus_mean(self, mu: np.ndarray) -> None:
        with torch.no_grad():
            self.mu_corp.copy_(torch.from_numpy(np.asarray(mu, dtype=np.float32).reshape(1, self.d_emb)).to(self.mu_corp.device))

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen unit vectors (..., d) -> adapted unit vectors e_hat (..., d)."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_emb).float()
        c_alpha = torch.clamp(self.center_alpha, 0.0, CENTER_MAX)
        if self.mu_corp.abs().sum() > 0 and c_alpha > 0:
            x_c = F.normalize(x_flat - c_alpha * self.mu_corp, p=2, dim=-1, eps=1e-8)
        else:
            x_c = F.normalize(x_flat, p=2, dim=-1, eps=1e-8)
        gate = torch.clamp(self.adapter_gate, 0.0, ADAPTER_GATE_MAX)
        delta = self.adapter_w2(F.gelu(self.adapter_w1(self.adapter_ln(x_c))))
        return F.normalize(x_c + gate * delta, p=2, dim=-1, eps=1e-8).reshape(orig_shape)

    def prototypes(self, h_supp: torch.Tensor, m_supp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adapted support papers (R, L, d) with mask (R, L) -> prototypes (R, M, d), their mask and masses pi (R, M)."""
        if self.proto_fp32 and h_supp.is_cuda:
            with torch.autocast("cuda", enabled=False):
                return self._prototypes(h_supp.float(), m_supp)
        return self._prototypes(h_supp, m_supp)

    def _prototypes(self, h_supp: torch.Tensor, m_supp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = h_supp.shape[2]
        z = self.in_proj(self.in_ln(h_supp))
        for block in self.set_blocks:
            z = block(z, m_supp)
        u_gate = torch.clamp(self.context_gate, 0.0, CONTEXT_GATE_MAX)
        h_ctx = F.normalize(h_supp + u_gate * self.out_proj(self.out_ln(z)), p=2, dim=-1, eps=1e-8) * m_supp.unsqueeze(-1).to(h_supp.dtype)
        seed_idx, pmask = _strided_seeds(m_supp, self.n_prototypes)
        seeds = torch.gather(h_ctx, 1, seed_idx.unsqueeze(-1).expand(-1, -1, D))
        sim = torch.bmm(h_ctx, seeds.transpose(1, 2)).masked_fill(~pmask.unsqueeze(1), -1e4)
        assign = F.softmax(sim / self.tau_a, dim=-1) * m_supp.unsqueeze(-1).to(h_ctx.dtype) * pmask.unsqueeze(1).to(h_ctx.dtype)
        mass = assign.sum(dim=1).clamp(min=1e-6) * pmask.to(h_ctx.dtype)
        pmass = mass / mass.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        protos_init = F.normalize(torch.bmm(assign.transpose(1, 2), h_ctx), p=2, dim=-1, eps=1e-8) * pmask.unsqueeze(-1).to(h_ctx.dtype)
        p_gate = torch.clamp(self.refine_gate, 0.0, REFINE_GATE_MAX)
        protos_delta = self.refine(protos_init, pmask, h_ctx, m_supp)
        protos = F.normalize(protos_init + p_gate * protos_delta, p=2, dim=-1, eps=1e-8) * pmask.unsqueeze(-1).to(h_ctx.dtype)
        return protos.float(), pmask, pmass.float()

    def core_scores(self, q_hat: torch.Tensor, protos: torch.Tensor, pmask: torch.Tensor, pmass: torch.Tensor,
                    pairwise: bool = False) -> torch.Tensor:
        """S_core. pairwise=False: every submission (B, d) against every reviewer (R, M, d) -> (B, R). pairwise=True: the
        p-th submission against the p-th reviewer's prototypes (P, M, d) -> (P,)."""
        qg = torch.clamp(self.query_gate, 0.0, QUERY_GATE_MAX)
        z_q = F.normalize(q_hat + qg * self.query_mlp(q_hat), p=2, dim=-1, eps=1e-8)
        if pairwise:
            sim = torch.einsum("pd,pmd->pm", z_q, protos)
            logits = (sim / self.tau_p + torch.log(pmass.clamp(min=1e-6))).masked_fill(~pmask, -1e4)
            s_core = self.tau_p * torch.logsumexp(logits, dim=-1)
            return s_core.masked_fill(~pmask.any(dim=-1), 0.0)
        sim = torch.einsum("bd,rmd->brm", z_q, protos)
        logits = (sim / self.tau_p + torch.log(pmass.clamp(min=1e-6)).unsqueeze(0)).masked_fill(~pmask.unsqueeze(0), -1e4)
        s_core = self.tau_p * torch.logsumexp(logits, dim=-1)
        return s_core.masked_fill(~pmask.any(dim=-1).unsqueeze(0), 0.0)

    def _fuse(self, s_core, s_niche, q_c_sim, log_len, coh):
        phi = torch.stack([q_c_sim, s_core - s_niche, log_len, coh], dim=-1)
        g_delta = self.fusion_mlp(self.fusion_ln(phi)).squeeze(-1)
        w_core = torch.sigmoid(self.core_logit + torch.clamp(self.fusion_scale, 0.0, FUSION_SCALE_MAX) * g_delta)
        return w_core * s_core + (1.0 - w_core) * s_niche

    # ------------------------------------------------------------------------------------------------------------------
    def score_all_pairs(self, q_raw: torch.Tensor, h_raw: torch.Tensor, h_mask: torch.Tensor,
                        h_supp_raw: torch.Tensor, h_supp_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Every submission q_raw (B, d) against every history h_raw (R, L, d) with mask (R, L), whose support sets are
        h_supp_raw (R, Ls, d) with mask (R, Ls). Frozen unit vectors in. Returns scores (B, R) and the training extras."""
        q_hat = self.adapt(q_raw)
        h_hat = self.adapt(h_raw) * h_mask.unsqueeze(-1).float()
        hs_hat = self.adapt(h_supp_raw) * h_supp_mask.unsqueeze(-1).float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=q_raw.is_cuda):
            protos, pmask, pmass = self.prototypes(hs_hat, h_supp_mask)
        s_core = self.core_scores(q_hat, protos, pmask, pmass)

        mf = h_mask.float()
        n_r = mf.sum(dim=-1).clamp(min=1.0)
        e_bar = F.normalize((h_hat * mf.unsqueeze(-1)).sum(dim=1), p=2, dim=-1, eps=1e-8)
        coh_r = ((h_hat * e_bar.unsqueeze(1)).sum(dim=-1) * mf).sum(dim=-1) / n_r

        beta = 0.50 * torch.sigmoid(self.beta_logit)
        e_tilde = F.normalize(h_hat - beta * e_bar.unsqueeze(1), p=2, dim=-1, eps=1e-8) * mf.unsqueeze(-1)
        ng = torch.clamp(self.niche_gate, 0.0, NICHE_GATE_MAX)
        q_n = F.normalize(q_hat + ng * self.niche_q(q_hat), p=2, dim=-1, eps=1e-8)
        k_n = F.normalize(e_tilde + ng * self.niche_k(e_tilde), p=2, dim=-1, eps=1e-8)
        v_n = F.normalize(h_hat + ng * self.niche_v(h_hat), p=2, dim=-1, eps=1e-8)
        sim_attn = torch.einsum("bd,rld->brl", q_n, k_n).masked_fill(~h_mask.unsqueeze(0), -1e4)
        attn = torch.softmax(sim_attn / self.tau_c, dim=-1) * mf.unsqueeze(0)
        z_rq = F.normalize(torch.einsum("brl,rld->brd", attn, v_n), p=2, dim=-1, eps=1e-8)
        s_niche = torch.einsum("bd,brd->br", q_hat, z_rq)

        q_c_sim = q_hat @ e_bar.T
        log_len = (torch.log1p(n_r) / LOG_LEN_NORM).unsqueeze(0).expand_as(s_core)
        s_fused = self._fuse(s_core, s_niche, q_c_sim, log_len, coh_r.unsqueeze(0).expand_as(s_core))
        offset = (self.offset_len * torch.log1p(n_r) + self.offset_coh * coh_r).unsqueeze(0)
        return {"scores": s_fused + offset, "s_core": s_core, "s_niche": s_niche, "protos": protos, "pmask": pmask,
                "h_hat": h_hat}

    @torch.no_grad()
    def score_items(self, X: np.ndarray, query_rows: np.ndarray, reviewer_keys: Sequence[str],
                    histories: Mapping[str, Sequence[int]], device: torch.device) -> np.ndarray:
        """Benchmark scoring. X (N, d) holds the frozen unit vectors of every paper of a benchmark, query_rows (P,) the
        submission row of each item, reviewer_keys (P,) its reviewer, and histories maps a reviewer to its paper rows
        (newest first). The support set is chosen on the adapted vectors, and the niche branch reads the whole history."""
        self.eval()
        N, D = X.shape
        X_t = torch.from_numpy(np.ascontiguousarray(X[:, :D], dtype=np.float32)).to(device)
        X_hat_t = torch.zeros((N, D), dtype=torch.float32, device=device)
        for s in range(0, N, 2048):
            e = min(N, s + 2048)
            X_hat_t[s:e] = self.adapt(X_t[s:e])
        X_hat_np = X_hat_t.cpu().numpy()

        unique_revs = list(dict.fromkeys(reviewer_keys))
        rev_to_idx = {r: i for i, r in enumerate(unique_revs)}
        R = len(unique_revs)
        supp_mat = torch.zeros((R, L_MAX_SCORE, D), dtype=torch.float32, device=device)
        supp_mask = torch.zeros((R, L_MAX_SCORE), dtype=torch.bool, device=device)
        for i, r in enumerate(unique_revs):
            rows = [int(x) for x in histories.get(r, ()) if 0 <= int(x) < N]
            if not rows:
                continue
            sel_rows = select_support_rows(rows, X=X_hat_np, l_max=L_MAX_SCORE, l_rec=L_REC_SCORE)
            if sel_rows:
                supp_mat[i, :len(sel_rows)] = X_hat_t[sel_rows]
                supp_mask[i, :len(sel_rows)] = True

        protos_all = torch.zeros((R, self.n_prototypes, D), dtype=torch.float32, device=device)
        pmask_all = torch.zeros((R, self.n_prototypes), dtype=torch.bool, device=device)
        pmass_all = torch.zeros((R, self.n_prototypes), dtype=torch.float32, device=device)
        for s in range(0, R, 128):
            e = min(R, s + 128)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pr, pm, pms = self.prototypes(supp_mat[s:e], supp_mask[s:e])
            protos_all[s:e], pmask_all[s:e], pmass_all[s:e] = pr, pm, pms

        groups: Dict[str, List[int]] = {}
        for pos_i, r in enumerate(reviewer_keys):
            groups.setdefault(r, []).append(pos_i)
        out_scores = np.zeros(len(query_rows), dtype=np.float32)
        beta = float((0.50 * torch.sigmoid(self.beta_logit)).item())
        ng = float(torch.clamp(self.niche_gate, 0.0, NICHE_GATE_MAX).item())
        for r, pos_list in groups.items():
            full_rows = [int(x) for x in histories.get(r, ()) if 0 <= int(x) < N]
            if not full_rows:
                continue
            pos = np.asarray(pos_list, dtype=np.int64)
            r_i = rev_to_idx[r]
            q_hat = X_hat_t[torch.from_numpy(query_rows[pos]).to(device)]
            n_p = len(pos)
            s_core = self.core_scores(q_hat, protos_all[r_i:r_i + 1].expand(n_p, -1, -1), pmask_all[r_i:r_i + 1].expand(n_p, -1),
                                      pmass_all[r_i:r_i + 1].expand(n_p, -1), pairwise=True)

            h_r = X_hat_t[torch.tensor(full_rows, dtype=torch.long, device=device)]
            n_r = float(len(full_rows))
            e_bar = F.normalize(h_r.mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-8)
            coh_r = float((h_r @ e_bar.T).mean().item())
            e_tilde = F.normalize(h_r - beta * e_bar, p=2, dim=-1, eps=1e-8)
            q_n = F.normalize(q_hat + ng * self.niche_q(q_hat), p=2, dim=-1, eps=1e-8)
            k_n = F.normalize(e_tilde + ng * self.niche_k(e_tilde), p=2, dim=-1, eps=1e-8)
            v_n = F.normalize(h_r + ng * self.niche_v(h_r), p=2, dim=-1, eps=1e-8)
            sim_attn = (q_n @ k_n.T) / self.tau_c
            sim_attn = sim_attn - sim_attn.max(dim=-1, keepdim=True).values
            z_rq = F.normalize(torch.softmax(sim_attn, dim=-1) @ v_n, p=2, dim=-1, eps=1e-8)
            s_niche = (q_hat * z_rq).sum(dim=-1)

            q_c_sim = (q_hat @ e_bar.T).squeeze(-1)
            s_fused = self._fuse(s_core, s_niche, q_c_sim, torch.full_like(s_core, math.log1p(n_r) / LOG_LEN_NORM),
                                 torch.full_like(s_core, coh_r))
            offset = float(self.offset_len.item()) * math.log1p(n_r) + float(self.offset_coh.item()) * coh_r
            out_scores[pos] = (s_fused + offset).cpu().numpy()
        return out_scores


def load_head(path, device: torch.device = torch.device("cpu")) -> MusterHead:
    """A released or trained head for scoring: FP32 prototypes, evaluation mode."""
    ckpt = torch.load(path, map_location=device)
    head = MusterHead(ModelConfig(**ckpt.get("model_config", {})), reference_rng=False).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.proto_fp32 = True
    return head.eval()
