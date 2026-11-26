#!/usr/bin/env python3
"""Frozen paper encodings (Section 3.1 of the paper): SPECTER2 base and SciNCL, FP32, at most 512 tokens.

corpus      training-corpus works (data/train/works.tsv) and the extra works of the long-history validation: text
            "title [SEP] abstract", multi-view pooling of the top layers, unit norm:
            v = norm(cls + 0.12 sal + 0.06 mean + 0.05 d2), with cls the last layer's [CLS] vector, mean the mean of its
            token vectors, sal their softmax-weighted sum by cosine to cls (temperature 0.15, [CLS] excluded) and
            d2 = norm(cls - layer-9 [CLS]), every view unit-normed. Writes vectors_<enc>.npy and mu_<enc>.npy (the mean
            of the rows in training authors' records) for data/train, and vectors_<enc>.npy for data/longval.
benchmarks  benchmark papers: the last layer's [CLS] vector, unit norm. Texts: SPECTER2 "title. abstract" everywhere;
            SciNCL "title[SEP]abstract" on LR-Bench and CMU and "title [SEP] abstract" on the ranking benchmarks.
check       re-encodes a random sample of every family and prints the row cosines to the released files.
The released vectors were encoded on GPUs (docs/DATA.md). A re-encoding on other hardware matches them to a row cosine
of about 0.9999, so a model trained or scored on re-encoded vectors differs slightly from the released results.

usage: python scripts/encode.py {corpus,benchmarks,check} --encoder sp2 [--out DIR] [--device cuda] [--sample 64]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muster.benchmarks import ENCODINGS, RANK_DATASETS, RAW  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = {"sp2": ("allenai/specter2_base", "3447645e1def9117997203454fa4495937bfbd83"),
          "sci": ("malteos/scincl", "ebc5348d184ba2fc9beee69b4e394263fce57b2e")}


def load_encoder(encoder: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer
    name, rev = MODELS[encoder]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    mdl = AutoModel.from_pretrained(name, revision=rev).to(device).eval()
    for p in mdl.parameters():
        p.requires_grad = False
    return mdl, tok


def pool_multiview(last: torch.Tensor, h9: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    cls_vec = F.normalize(last[:, 0].float(), p=2, dim=-1, eps=1e-8)
    mask_f = attention_mask.unsqueeze(-1).float()
    mean_vec = F.normalize((last.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0), p=2, dim=-1, eps=1e-8)
    tok_norm = F.normalize(last.float(), p=2, dim=-1, eps=1e-8)
    valid = attention_mask.bool().clone()
    valid[:, 0] = False
    sal = torch.einsum("bd,bld->bl", cls_vec, tok_norm).masked_fill(~valid, -1e4)
    sal_w = torch.softmax(sal / 0.15, dim=-1) * valid.float()
    sal_vec = F.normalize(torch.einsum("bl,bld->bd", sal_w, last.float()), p=2, dim=-1, eps=1e-8)
    d2 = F.normalize(cls_vec - F.normalize(h9[:, 0].float(), p=2, dim=-1, eps=1e-8), p=2, dim=-1, eps=1e-8)
    return F.normalize(cls_vec + 0.12 * sal_vec + 0.06 * mean_vec + 0.05 * d2, p=2, dim=-1, eps=1e-8)


@torch.no_grad()
def encode(mdl, tok, texts: List[str], device: torch.device, pooling: str, bs: int = 64) -> np.ndarray:
    """Batches of similar length (dynamic padding); returns float32 unit vectors in input order."""
    out = np.zeros((len(texts), 768), dtype=np.float32)
    order = np.argsort(np.asarray([len(t) for t in texts]), kind="stable")
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        enc = tok([texts[int(i)] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt",
                  return_token_type_ids=False).to(device)
        o = mdl(**enc, output_hidden_states=pooling == "multiview")
        if pooling == "multiview":
            v = pool_multiview(o.last_hidden_state, o.hidden_states[10], enc["attention_mask"])
        else:
            v = F.normalize(o.last_hidden_state[:, 0].float(), dim=-1)
        out[idx] = v.float().cpu().numpy()
    return out


def corpus_texts(lines: List[int], sep: str, corpus: Path = REPO / "data" / "corpus" / "corpus_elite_ccf_core_thu.jsonl.gz"):
    want = set(lines)
    txt: Dict[int, str] = {}
    with gzip.open(corpus, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                txt[i] = f"{(r.get('title') or '').strip()} {sep} {(r.get('abstract') or '').strip()}"
    return [txt[i] for i in lines]


def works(name: str):
    rows = [line.split("\t") for line in (REPO / "data" / name / "works.tsv").read_text().splitlines()[1:]]
    return [int(r[2]) for r in rows], [int(r[3]) for r in rows] if name == "longval" else None


def benchmark_texts(encoder: str, sep: str) -> Dict[str, List[str]]:
    dot = lambda t, a: f"{t}. {a}".strip()  # noqa: E731
    pref = dot if encoder == "sp2" else (lambda t, a: f"{t}{sep}{a}")
    rank = dot if encoder == "sp2" else (lambda t, a: f"{t} {sep} {a}".strip())
    lr: Dict[str, tuple] = {}
    for fn in ("evaluations_pc.json", "evaluations_rc.json"):
        for ex in json.load(open(RAW / "lr_bench" / fn)):
            for side in (ex["anchor"], ex["positive"], ex["negative"]):
                if "paper_id" in side:
                    lr[str(side["paper_id"])] = (side.get("paper_title", ""), side.get("abstract", ""))
                if "author_id" in side:
                    for i, p in enumerate(side.get("papers", [])):
                        lr[f"{side['author_id']}#{i}"] = (p.get("paper_title") or p.get("title", ""), p.get("abstract", ""))
    gold: Dict[str, tuple] = {}
    for d in sorted((RAW / "cmu_gold" / "evaluation_datasets").glob("d_20_*")):
        for pid, rec in json.load(open(d / "submissions.json")).items():
            c = rec.get("content", {})
            gold.setdefault(str(pid), (c.get("title", ""), c.get("abstract", "")))
        for fa in sorted((d / "archives").glob("*.jsonl")):
            for line in open(fa):
                if line.strip():
                    rec = json.loads(line)
                    c = rec.get("content", {})
                    gold.setdefault(str(rec["id"]), (c.get("title", ""), c.get("abstract", "")))
    ids = {k: json.load(open(ENCODINGS / "ids" / f"{k}.json")) for k in ("lr", "cmu") + RANK_DATASETS}
    out = {"lr": [pref(*lr.get(str(p), ("", ""))) for p in ids["lr"]],
           "cmu": [pref(*gold.get(str(p), ("", ""))) for p in ids["cmu"]]}
    for ds in RANK_DATASETS:
        from muster.benchmarks import _json
        papers = {}
        for p in _json(RAW / ds / f"{ds}_papers_test.json"):
            papers.setdefault(str(p["paper"]), (p.get("title") or "", p.get("abstract") or ""))
        out[ds] = [rank(*papers[str(p)]) for p in ids[ds]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["corpus", "benchmarks", "check"])
    ap.add_argument("--encoder", choices=sorted(MODELS), required=True)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "encodings", help="output directory (not the released files)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample", type=int, default=64)
    a = ap.parse_args()
    device = torch.device(a.device)
    mdl, tok = load_encoder(a.encoder, device)
    sep = tok.sep_token or "[SEP]"
    if a.mode == "corpus":
        lines, _ = works("train")
        V = encode(mdl, tok, corpus_texts(lines, sep), device, "multiview")
        seq_idx = np.load(REPO / "data" / "train" / "episodes.npz")["seq_idx"]
        is_train = np.zeros(len(V), dtype=bool)
        is_train[seq_idx] = True
        (a.out / "train").mkdir(parents=True, exist_ok=True)
        np.save(a.out / "train" / f"vectors_{a.encoder}.npy", V)
        np.save(a.out / "train" / f"mu_{a.encoder}.npy", V[is_train].mean(0, keepdims=True).astype(np.float32))
        lv_lines, lv_train_row = works("longval")
        W = np.zeros((len(lv_lines), 768), dtype=np.float32)
        have = np.asarray(lv_train_row) >= 0
        W[have] = V[np.asarray(lv_train_row)[have]]
        W[~have] = encode(mdl, tok, corpus_texts([l for l, h in zip(lv_lines, have) if not h], sep), device, "multiview")
        (a.out / "longval").mkdir(parents=True, exist_ok=True)
        np.save(a.out / "longval" / f"vectors_{a.encoder}.npy", W)
        print("wrote", a.out / "train", a.out / "longval")
    elif a.mode == "benchmarks":
        (a.out / a.encoder).mkdir(parents=True, exist_ok=True)
        for k, texts in benchmark_texts(a.encoder, sep).items():
            np.save(a.out / a.encoder / f"{k}.npy", encode(mdl, tok, texts, device, "cls"))
            print("wrote", a.out / a.encoder / f"{k}.npy", flush=True)
    else:
        rng = np.random.default_rng(0)
        lines, _ = works("train")
        rel = np.load(REPO / "data" / "train" / f"vectors_{a.encoder}.npy", mmap_mode="r")
        pick = np.sort(rng.choice(len(lines), size=a.sample, replace=False))
        V = encode(mdl, tok, corpus_texts([lines[i] for i in pick], sep), device, "multiview")
        report = {"corpus": V}
        refs = {"corpus": np.asarray(rel[pick])}
        bt = benchmark_texts(a.encoder, sep)
        for k, texts in bt.items():
            R = np.load(ENCODINGS / a.encoder / f"{k}.npy", mmap_mode="r")
            active = np.nonzero(np.abs(np.asarray(R)).sum(1) > 0)[0]  # SciNCL ranking files keep unused papers at zero
            p = np.sort(rng.choice(active, size=min(a.sample, len(active)), replace=False))
            report[k], refs[k] = encode(mdl, tok, [texts[i] for i in p], device, "cls"), np.asarray(R[p])
        for k in report:
            c = (report[k] * refs[k]).sum(1) / np.linalg.norm(refs[k], axis=1)
            print(f"{a.encoder} {k:10s} n {len(c):3d}  row cosine to the released vectors: mean {c.mean():.6f}  min {c.min():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
