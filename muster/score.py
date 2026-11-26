"""Score the seven benchmarks with MUSTER heads and combine the two encoders."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch

from .benchmarks import RANK_DATASETS, PreferenceBenchmarks, RankingBenchmark
from .config import W_SCI
from .model import MusterHead


def score_benchmarks(head: MusterHead, enc: Mapping[str, np.ndarray], pref: PreferenceBenchmarks,
                     ranking: Mapping[str, RankingBenchmark], device: torch.device) -> Dict[str, object]:
    """Scores (float64) of every benchmark item: lr (897,), gold (10, 477) and rank[ds] aligned with ranking[ds].pairs."""
    it = pref.lr_items
    lr = head.score_items(enc["lr"], it.query_rows, it.reviewer_keys, it.histories, device)
    gold = np.stack([head.score_items(enc["cmu"], g.query_rows, g.reviewer_keys, g.histories, device) for g in pref.gold_items])
    rank = {ds: head.score_items(enc[ds], ranking[ds].items.query_rows, ranking[ds].items.reviewer_keys,
                                 ranking[ds].items.histories, device) for ds in RANK_DATASETS}
    return {"lr": np.asarray(lr, np.float64), "gold": np.asarray(gold, np.float64),
            "rank": {ds: np.asarray(rank[ds], np.float64) for ds in RANK_DATASETS}}


def combine(s_sp2: Mapping[str, object], s_sci: Mapping[str, object], w_sci: float = W_SCI) -> Dict[str, object]:
    """Dual-backbone average: (1 - w_sci) * SPECTER2 scores + w_sci * SciNCL scores."""
    mix = lambda a, b: (1.0 - w_sci) * a + w_sci * b  # noqa: E731
    return {"lr": mix(s_sp2["lr"], s_sci["lr"]), "gold": mix(s_sp2["gold"], s_sci["gold"]),
            "rank": {ds: mix(s_sp2["rank"][ds], s_sci["rank"][ds]) for ds in RANK_DATASETS}}


def save_scores(path, s: Mapping[str, object]) -> None:
    np.savez_compressed(path, lr=s["lr"], gold=s["gold"], **{f"rank_{ds}": s["rank"][ds] for ds in RANK_DATASETS})


def load_scores(path) -> Dict[str, object]:
    z = np.load(path)
    return {"lr": z["lr"], "gold": z["gold"], "rank": {ds: z[f"rank_{ds}"] for ds in RANK_DATASETS}}
