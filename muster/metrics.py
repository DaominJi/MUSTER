"""Metrics of the paper's tables.

Preference (per track): accuracy = share of comparisons ordered correctly (a score tie counts as wrong), loss = weighted
share ordered wrongly with weight |td| (a tie counts half). CMU reports 100 (1 - loss) as its accuracy, averaged over the
10 profile samples, and the unweighted accuracy over all samples' comparisons. The macro averages the three rounded
track values.
Ranking (per benchmark and split): for each query, rank its pool by score (ties broken by the reviewer id as a string)
and read P@k = relevant reviewers in the top k divided by k, and nDCG@3 with gain 2^rel - 1. A query without a relevant
reviewer counts as 0. The macro averages the four benchmarks.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from .benchmarks import RANK_DATASETS


def preference_metrics(tracks: Mapping[str, Mapping[str, np.ndarray]], lr_scores: np.ndarray,
                       gold_scores: np.ndarray) -> Dict[str, float]:
    """lr_scores (897,) aligned with the LR items, gold_scores (10, 477) aligned with the CMU items, as float64."""
    out: Dict[str, float] = {}
    for tag in ("pc", "rc"):
        t = tracks[tag]
        prod = (lr_scores[t["a"]] - lr_scores[t["b"]]) * t["td"]
        out[f"{tag}_acc"] = round(100.0 * float(np.mean(prod > 0)), 2)
        out[f"{tag}_loss"] = round(
            float(np.sum(np.where(prod < 0, t["w"], np.where(prod == 0, 0.5 * t["w"], 0.0))) / np.sum(t["w"])), 4)
    g = tracks["cmu"]
    g_prod = (gold_scores[:, g["a"]] - gold_scores[:, g["b"]]) * g["td"][None, :]
    g_losses = np.sum(np.where(g_prod < 0, g["w"][None, :], np.where(g_prod == 0, 0.5 * g["w"][None, :], 0.0)),
                      axis=-1) / float(np.sum(g["w"]))
    cmu_loss = float(np.mean(g_losses))
    out["cmu_acc"] = round(100.0 * (1.0 - cmu_loss), 2)
    out["cmu_unweighted_acc"] = round(100.0 * float(np.mean(g_prod > 0)), 2)
    out["cmu_loss"] = round(cmu_loss, 4)
    out["pref_mean_acc"] = round((out["pc_acc"] + out["rc_acc"] + out["cmu_acc"]) / 3.0, 2)
    out["pref_mean_unweighted_acc"] = round((out["pc_acc"] + out["rc_acc"] + out["cmu_unweighted_acc"]) / 3.0, 2)
    out["pref_mean_loss"] = round((out["pc_loss"] + out["rc_loss"] + out["cmu_loss"]) / 3.0, 4)
    return out


def _ndcg_at_k(rels_ranked: np.ndarray, k: int) -> float:
    kk = min(k, len(rels_ranked))
    if kk <= 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, kk + 2))
    ideal = ((2.0 ** np.sort(rels_ranked)[::-1][:kk] - 1.0) * disc).sum()
    if ideal <= 0.0:
        return 0.0
    return float(((2.0 ** rels_ranked[:kk] - 1.0) * disc).sum() / ideal)


def ranking_metrics_one(queries: Mapping[str, list], pair_scores: Mapping[Tuple[str, str], float]) -> Dict[str, float]:
    """S-P@5, S-P@10, H-P@5, H-P@10 and S-N@3, H-N@3 (in %) of one benchmark from its item scores {(qid, rid): s}."""
    res: Dict[str, float] = {}
    for split in ("soft", "hard"):
        p5, p10, n3 = [], [], []
        for q in queries[split]:
            qid, labels = str(q["query_id"]), q["score"]
            rk = sorted(labels, key=lambda r: (-pair_scores.get((qid, str(r)), float("-inf")), str(r)))
            rels = np.array([float(labels[r]) for r in rk], dtype=np.float64)
            if rels.sum() == 0:
                p5.append(0.0), p10.append(0.0), n3.append(0.0)
                continue
            p5.append(rels[:5].sum() / 5)
            p10.append(rels[:10].sum() / 10)
            n3.append(_ndcg_at_k(rels, 3))
        s = split[0]
        res[f"{s}p5"], res[f"{s}p10"] = 100.0 * float(np.mean(p5)), 100.0 * float(np.mean(p10))
        res[f"{s}n3"] = float(100.0 * np.mean(n3))
    return res


def ranking_metrics(benches: Mapping[str, object], scores: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Per benchmark and the macro over NIPS, SIGIR, SciRepEval and KDD. scores[ds] aligned with benches[ds].pairs."""
    per = {}
    for ds in RANK_DATASETS:
        sc = np.asarray(scores[ds], dtype=np.float64)
        pair = {p: float(sc[i]) for i, p in enumerate(benches[ds].pairs)}
        per[ds] = ranking_metrics_one(benches[ds].queries, pair)
    per["macro"] = {k: float(np.mean([per[ds][k] for ds in RANK_DATASETS])) for k in per[RANK_DATASETS[0]]}
    return per
