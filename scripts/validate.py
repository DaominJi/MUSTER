#!/usr/bin/env python3
"""Benchmark-free validation metrics of a trained head: validation-suite composite and long-history MRR.

usage: python scripts/validate.py --encoder sp2 [--head checkpoints/muster_sp2.pt] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muster.config import ENCODERS  # noqa: E402
from muster.data import checkpoint_path, load_long_validation, load_train_pack  # noqa: E402
from muster.model import load_head  # noqa: E402
from muster.validate import long_history, validation_suite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--head", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="write the metrics to this JSON file")
    a = ap.parse_args()
    device = torch.device(a.device)
    head = load_head(a.head or checkpoint_path(a.encoder), device)
    pack = load_train_pack(a.encoder)
    vs = validation_suite(head, torch.from_numpy(pack["vecs_train"]).to(device), pack, device)
    vs.pop("_rr")
    vs["long_history"] = long_history(head, load_long_validation(a.encoder), device)
    print(json.dumps(vs, indent=1))
    if a.out:
        a.out.write_text(json.dumps(vs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
