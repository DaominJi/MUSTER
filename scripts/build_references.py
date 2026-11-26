#!/usr/bin/env python3
"""Build the screen's reference index (data/screen/refs_csr.npz) from the OpenAlex reference lists of the corpus rows.

data/screen/references.jsonl.gz holds one line per row of data/train/works.tsv: {"id", "refs"} with the referenced
OpenAlex works as returned on 28 Oct 2026, or {"id", "returned": false} for a work OpenAlex did not return. A row has
usable references (has_refs) when its list is not empty. Rows without a usable list make the citation evidence G
missing, never "no shared references". Column ids number the referenced works in order of first appearance.

usage: python scripts/build_references.py [--check]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCREEN = REPO / "data" / "screen"


def build(refs_path: Path, works_path: Path):
    ids = [line.split("\t")[1] for line in works_path.read_text().splitlines()[1:]]
    refs = {}
    with gzip.open(refs_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            refs[r["id"]] = r.get("refs", [])
    assert all(w in refs for w in ids), "a corpus row has no line in the reference file"
    vocab: dict = {}
    indptr = np.zeros(len(ids) + 1, dtype=np.int64)
    cols = []
    for i, w in enumerate(ids):
        rs = sorted({vocab.setdefault(x, len(vocab)) for x in refs[w]})
        cols.extend(rs)
        indptr[i + 1] = indptr[i] + len(rs)
    has = np.diff(indptr) > 0
    return {"indptr": indptr, "indices": np.asarray(cols, dtype=np.int32), "has_refs": has, "n_vocab": np.int64(len(vocab))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=Path, default=SCREEN / "references.jsonl.gz")
    ap.add_argument("--works", type=Path, default=REPO / "data" / "train" / "works.tsv")
    ap.add_argument("--out", type=Path, default=SCREEN / "refs_csr.npz")
    ap.add_argument("--check", action="store_true", help="compare with the existing index instead of writing it")
    a = ap.parse_args()
    arrays = build(a.refs, a.works)
    print(f"{len(arrays['indptr']) - 1:,} rows, {int(arrays['has_refs'].sum()):,} with references, "
          f"{int(arrays['n_vocab']):,} referenced works")
    if a.check:
        z = np.load(a.out)
        ok = set(z.files) == set(arrays) and all(z[k].dtype == v.dtype and np.array_equal(z[k], v) for k, v in arrays.items())
        print("CHECK", "OK: equals" if ok else "FAILED: differs from", a.out)
        return 0 if ok else 1
    np.savez(a.out, **arrays)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
