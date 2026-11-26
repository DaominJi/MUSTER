#!/usr/bin/env python3
"""Train one MUSTER head on the frozen embeddings of one encoder, then run validation on the held-out split.

usage: python scripts/train.py --encoder sp2 --out runs/sp2_s42 [--seed 42] [--steps 4218] [--device cuda]
Writes OUT/head.pt (readable by muster.model.load_head and scripts/evaluate.py), OUT/train_info.json, and
OUT/validation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muster.config import ENCODERS, TrainConfig  # noqa: E402
from muster.data import load_long_validation, load_train_pack, screen_files  # noqa: E402
from muster.train import train_head  # noqa: E402
from muster.validate import long_history, validation_suite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", type=Path, default=None, help="data directory (default: the repository's data/)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    logf = open(a.out / "log.txt", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] [{a.encoder}/s{a.seed}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    kw = {} if a.data is None else {"data_dir": a.data}
    cfg = TrainConfig(seed=a.seed, steps=a.steps)
    pack = load_train_pack(a.encoder, **kw)
    model_path, refs_path = screen_files(a.encoder, **kw)
    log(f"training {ENCODERS[a.encoder]} head on {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")
    head, info = train_head(pack, model_path, refs_path, cfg, device, log)
    torch.save({"state_dict": head.state_dict(), "model_config": asdict(cfg.model), "encoder": a.encoder,
                "train_config": cfg.to_dict()}, a.out / "head.pt")
    (a.out / "train_info.json").write_text(json.dumps(info))

    head.proto_fp32 = True
    vecs_gpu = torch.from_numpy(pack["vecs_train"]).to(device)
    vs = validation_suite(head, vecs_gpu, pack, device)
    np.savez_compressed(a.out / "val_rr.npz", **vs.pop("_rr"))
    vs["long_history"] = long_history(head, load_long_validation(a.encoder, **kw), device)
    (a.out / "validation.json").write_text(json.dumps(vs, indent=1))
    log(f"validation: composite {vs['composite']}, long-history MRR (> 48 papers) {vs['long_history']['mrr_long_all_gt48']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
