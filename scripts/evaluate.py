#!/usr/bin/env python3
"""Score the seven benchmarks with the SPECTER2 and SciNCL heads and their dual-backbone average.

usage: python scripts/evaluate.py [--sp2 checkpoints/muster_sp2.pt] [--sci checkpoints/muster_sci.pt]
                                  [--out runs/eval] [--device cpu]
Preference: accuracy per track (LR-PC, LR-RC, CMU [unweighted]) and loss, macro over the three tracks.
Ranking: soft and hard P@5 and P@10 per benchmark and their macro average.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muster.benchmarks import RANK_DATASETS, load_encodings, load_preference, load_ranking  # noqa: E402
from muster.data import checkpoint_path  # noqa: E402
from muster.metrics import preference_metrics, ranking_metrics  # noqa: E402
from muster.model import load_head  # noqa: E402
from muster.score import combine, save_scores, score_benchmarks  # noqa: E402

NAMES = {"sp2": "MUSTER (SPECTER2)", "sci": "MUSTER (SciNCL)", "default": "MUSTER (Default)"}
PREF_KEYS = ["pc_acc", "pc_loss", "rc_acc", "rc_loss", "cmu_acc", "cmu_unweighted_acc", "cmu_loss", "pref_mean_acc",
             "pref_mean_unweighted_acc", "pref_mean_loss"]
RANK_KEYS = ["sp5", "hp5", "sp10", "hp10"]


def summarize(s, pref, ranking):
    pm = preference_metrics(pref.tracks, s["lr"], s["gold"])
    rm = ranking_metrics(ranking, s["rank"])
    return {"preference": {k: pm[k] for k in PREF_KEYS},
            "ranking": {ds: {k: rm[ds][k] for k in RANK_KEYS} for ds in RANK_DATASETS + ("macro",)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp2", type=Path, default=checkpoint_path("sp2"))
    ap.add_argument("--sci", type=Path, default=checkpoint_path("sci"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="optional directory for saving score arrays and metrics")
    a = ap.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device(a.device)
    pref = load_preference()
    ranking = {ds: load_ranking(ds) for ds in RANK_DATASETS}
    scores = {}
    for enc_name, ckpt in (("sp2", a.sp2), ("sci", a.sci)):
        t0 = time.time()
        scores[enc_name] = score_benchmarks(load_head(ckpt, device), load_encodings(enc_name), pref, ranking, device)
        print(f"{NAMES[enc_name]}: scored in {time.time() - t0:.0f}s", flush=True)
    scores["default"] = combine(scores["sp2"], scores["sci"])
    res = {k: summarize(s, pref, ranking) for k, s in scores.items()}
    for k in ("sp2", "sci", "default"):
        p, r = res[k]["preference"], res[k]["ranking"]
        print(f"\n{NAMES[k]}")
        print(f"  LR-PC {p['pc_acc']:.2f} ({p['pc_loss']:.4f})  LR-RC {p['rc_acc']:.2f} ({p['rc_loss']:.4f})  "
              f"CMU {p['cmu_acc']:.2f} [{p['cmu_unweighted_acc']:.2f}] ({p['cmu_loss']:.4f})  "
              f"macro {p['pref_mean_acc']:.2f} [{p['pref_mean_unweighted_acc']:.2f}] ({p['pref_mean_loss']:.4f})")
        for ds in RANK_DATASETS + ("macro",):
            print(f"  {ds:10s} S-P@5 {r[ds]['sp5']:.2f}  H-P@5 {r[ds]['hp5']:.2f}  "
                  f"S-P@10 {r[ds]['sp10']:.2f}  H-P@10 {r[ds]['hp10']:.2f}")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        for k, s in scores.items():
            save_scores(a.out / f"scores_{k}.npz", s)
        (a.out / "metrics.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
