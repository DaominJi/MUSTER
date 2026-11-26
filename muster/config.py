"""Default settings of MUSTER.

Every value here is the one used to train and evaluate the released heads. The model structure lives in `ModelConfig`
and the training recipe in `TrainConfig`, so a run's full setting can be saved next to its checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

ENCODERS = {"sp2": "allenai/specter2_base", "sci": "malteos/scincl"}  # frozen paper encoders, one head each
D_EMB = 768  # width of both encoders' embeddings

# Support sets. A support set keeps every paper of a history of at most l_max papers. Otherwise it keeps the l_rec newest
# papers and one representative paper from each of l_max - l_rec consecutive chronological segments of the older papers.
L_MAX_SCORE, L_REC_SCORE = 64, 32  # scoring (benchmarks, long-history validation)
L_MAX_TRAIN, L_REC_TRAIN = 32, 16  # training episodes and checkpoint validation
LAMBDA_RED = 0.25  # weight of the redundancy to the newest papers in the representative choice

# Teacher score s0 on the frozen embeddings: GAMMA_TOP1 * top-1 + GAMMA_TOP3 * mean of the top-3 cosine similarities.
GAMMA_TOP1, GAMMA_TOP3 = 0.55, 0.45

W_SCI = 0.5  # MUSTER (Default): score = (1 - W_SCI) * SPECTER2 head + W_SCI * SciNCL head


@dataclass
class ModelConfig:
    d_emb: int = D_EMB
    n_blocks: int = 4  # pre-norm set attention blocks of the core branch
    n_heads: int = 8
    ffn_mult: int = 4
    n_prototypes: int = 12  # M
    adapter_hidden: int = 384
    niche_rank: int = 128  # rank of the low-rank residual maps W_Q, W_K, W_V of the niche branch
    center_init: float = 0.35  # initial centering strength alpha_c
    beta_init: float = 0.20  # initial distinctiveness beta
    tau_c: float = 0.04  # niche attention temperature
    tau_p: float = 0.08  # smooth-maximum temperature of the core score
    tau_a: float = 0.08  # temperature of the soft assignment to the prototype seeds
    omega_init: float = 0.25  # initial weight of the core score in the fusion, 1 - omega in the paper's notation


@dataclass
class TrainConfig:
    seed: int = 42
    steps: int = 6 * (90000 // 128)  # 4,218: six passes over the 90,000 training episodes
    batch: int = 128
    lr: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.06  # warmup = max(60, warmup_frac * steps), then cosine decay to lr_floor * lr
    lr_floor: float = 0.10
    clip: float = 1.0
    ema: float = 0.995
    val_every: int = 400  # checkpoint validation on the EMA head; the best of steps >= val_every is kept
    val_episodes: int = 500
    amp_bf16: bool = True  # BF16 autocast for the training forward pass (FP32 master weights, FP32 validation)
    p0: float = 0.25  # initial competence of the curriculum
    lam_rc: float = 1.0  # weights of the reviewer-centric, distillation, coverage and diversity losses
    lam_kd: float = 0.30
    lam_cov: float = 0.05
    lam_div: float = 0.05
    tau_kd: float = 0.10
    div_margin: float = 0.22  # prototype pairs with cosine above this are pushed apart
    screen_delta: float = 0.2  # a non-author stays a negative if its posterior of being qualified is at most delta
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
