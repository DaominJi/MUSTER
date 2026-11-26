#!/usr/bin/env python3
"""Build the training and validation episodes from the corpus (CPU, about a minute).

Corpus: data/corpus/corpus_elite_ccf_core_thu.jsonl.gz, one paper per line (OpenAlex id, title, abstract, authors, year,
venue), papers of CCF A/B, CORE A*/A and THU A/B venues, with every paper of the seven benchmarks and every benchmark
reviewer removed (docs/DATA.md).

Steps, as in the code that built the released training data:
  1 author records: each paper credits its first four authors; a paper is keyed by its line number, and its text is
    "title [SEP] abstract"
  2 authors are ranked by their number of credited papers (ties in order of first appearance); an author's papers are
    sorted by (integer year, line number as a string) and a paper whose lowercased first 64 characters of text repeat an
    earlier one is dropped; authors with at least 6 papers are kept, each with its newest 49, until 32,000 authors
  3 a seeded permutation (numpy default_rng(42)) splits them into 30,000 training and 2,000 validation authors
  4 the corpus rows are the union of the kept papers, ordered by the string order of their line numbers (446,314 rows)
  5 training episodes: for each training author, its 3 newest papers each form a query, with the author's up to 48
    previous papers, newest first, as the history (90,000 episodes); validation: each validation author's newest paper
    with up to 48 previous papers
  6 long-history validation: the same 2,000 validation authors with their full earlier records (up to 256 papers)
Writes data/train/episodes.npz and works.tsv, data/longval/episodes.npz and works.tsv. With --check it compares the
arrays with the existing files instead of writing them.

usage: python scripts/build_training_data.py [--corpus PATH] [--out data] [--check]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
N_TRAIN, N_VAL, MIN_PAPERS, MAX_PAPERS, CUTOFFS, MAX_HIST, H_MAX_LONG, SEED = 30000, 2000, 6, 49, 3, 48, 256, 42


def csr(lists: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(lists) + 1, dtype=np.int64)
    for i, l in enumerate(lists):
        indptr[i + 1] = indptr[i] + len(l)
    return indptr, np.fromiter((x for l in lists for x in l), dtype=np.int32, count=int(indptr[-1]))


def co_author_sets(work_lists: List[set]) -> Tuple[np.ndarray, np.ndarray]:
    """For each author, the other authors (of the same list) that share a paper with it, as CSR."""
    by_work: Dict[int, List[int]] = {}
    for i, ws in enumerate(work_lists):
        for w in ws:
            by_work.setdefault(w, []).append(i)
    co = [set() for _ in work_lists]
    for al in by_work.values():
        if len(al) > 1:
            for x in al:
                co[x].update(al)
    for i in range(len(co)):
        co[i].discard(i)
    return csr([sorted(s) for s in co])


def build(corpus: Path):
    t0 = time.time()
    by_author: Dict[str, List[Tuple[int, str]]] = {}
    work_texts: Dict[str, str] = {}
    paper_id: Dict[str, str] = {}
    with gzip.open(corpus, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            aids_raw = rec.get("authors")
            if not (isinstance(aids_raw, list) and aids_raw):
                continue
            aids = [str(a) for a in aids_raw[:4] if a]
            wid = str(idx)
            yr = int(float(rec.get("year") or 2020))
            title, abstract = (rec.get("title") or "").strip(), (rec.get("abstract") or "").strip()
            if len(title) < 10 or len(abstract) < 40:
                continue
            if wid not in work_texts:
                work_texts[wid] = f"{title} [SEP] {abstract}"
                paper_id[wid] = str(rec.get("paper_id"))
            for aid in aids:
                by_author.setdefault(aid, []).append((yr, wid))
    full: List[List[str]] = []  # each kept author's whole deduplicated record, oldest first
    for aid, items in sorted(by_author.items(), key=lambda kv: -len(kv[1])):
        if len(items) < MIN_PAPERS:
            continue
        items.sort(key=lambda x: (x[0], x[1]))
        dedup, seen = [], set()
        for _, wid in items:
            prefix = work_texts[wid][:64].lower()
            if prefix not in seen:
                seen.add(prefix)
                dedup.append(wid)
        if len(dedup) >= MIN_PAPERS:
            full.append(dedup)
            if len(full) >= N_TRAIN + N_VAL:
                break
    trunc = [s[-MAX_PAPERS:] for s in full]
    perm = np.random.default_rng(SEED).permutation(len(full))
    n_train = min(N_TRAIN, max(1, len(full) - N_VAL))
    n_val = min(N_VAL, max(0, len(full) - n_train))
    tr_ids, va_ids = perm[:n_train], perm[n_train:n_train + n_val]
    train_wids = [trunc[int(i)] for i in tr_ids]
    val_wids = [trunc[int(i)] for i in va_ids]
    active = sorted({w for s in train_wids + val_wids for w in s})
    w2i = {w: i for i, w in enumerate(active)}
    seqs = [[w2i[w] for w in s] for s in train_wids]
    vseqs = [[w2i[w] for w in s] for s in val_wids]

    q_tr, h_tr, ep_author = [], [], []
    for ai, seq in enumerate(seqs):
        for c in range(CUTOFFS):
            q_pos = len(seq) - 1 - c
            if q_pos < 4:
                break
            q_tr.append(seq[q_pos])
            h_tr.append(list(reversed(seq[max(0, q_pos - MAX_HIST):q_pos])))
            ep_author.append(ai)
    q_va = [s[-1] for s in vseqs]
    h_va = [list(reversed(s[:-1]))[:MAX_HIST] for s in vseqs]

    def pad(hists, width):
        h = np.zeros((len(hists), width), dtype=np.int32)
        m = np.zeros((len(hists), width), dtype=bool)
        for i, hh in enumerate(hists):
            h[i, :len(hh)] = hh
            m[i, :len(hh)] = True
        return h, m
    train_h_idx, train_m = pad(h_tr, MAX_HIST)
    val_h_idx, val_m = pad(h_va, MAX_HIST)
    seq_indptr, seq_idx = csr(seqs)
    vco_indptr, vco_idx = co_author_sets([set([int(q_va[i])] + [int(x) for x in val_h_idx[i][val_m[i]]]) for i in range(len(q_va))])
    train = {"train_q_idx": np.asarray(q_tr, dtype=np.int32), "train_h_idx": train_h_idx, "train_m": train_m,
             "ep_author": np.asarray(ep_author, dtype=np.int32), "seq_indptr": seq_indptr, "seq_idx": seq_idx,
             "val_q_idx": np.asarray(q_va, dtype=np.int32), "val_h_idx": val_h_idx, "val_m": val_m,
             "vco_indptr": vco_indptr, "vco_idx": vco_idx}

    val_full = [full[int(i)] for i in va_ids]
    need = sorted({w for s in val_full for w in s})
    lw2i = {w: i for i, w in enumerate(need)}
    lh, lm = pad([[lw2i[w] for w in reversed(s[:-1])][:H_MAX_LONG] for s in val_full], H_MAX_LONG)
    longval = {"q": np.asarray([lw2i[s[-1]] for s in val_full], dtype=np.int32), "h": lh, "m": lm,
               "val_vco_indptr": vco_indptr, "val_vco_idx": vco_idx}
    works = [(i, paper_id[w], int(w)) for i, w in enumerate(active)]
    long_works = [(i, paper_id[w], int(w), w2i.get(w, -1)) for i, w in enumerate(need)]
    print(f"{len(full):,} authors ({n_train:,} training, {n_val:,} validation), {len(active):,} corpus rows, "
          f"{len(q_tr):,} training episodes, {len(need):,} long-validation works ({time.time() - t0:.0f}s)", flush=True)
    return train, longval, works, long_works


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus" / "corpus_elite_ccf_core_thu.jsonl.gz")
    ap.add_argument("--out", type=Path, default=REPO / "data")
    ap.add_argument("--check", action="store_true", help="compare with the existing files instead of writing")
    a = ap.parse_args()
    train, longval, works, long_works = build(a.corpus)
    if a.check:
        bad = []
        for name, arrays in (("train", train), ("longval", longval)):
            z = np.load(a.out / name / "episodes.npz")
            for k, v in arrays.items():
                if not (z[k].dtype == v.dtype and np.array_equal(z[k], v)):
                    bad.append(f"{name}/{k}")
        for name, rows in (("train", works), ("longval", long_works)):
            p = a.out / name / "works.tsv"
            if not p.exists():
                print(f"({p} not there yet, not compared)")
                continue
            have = [tuple(int(x) if j != 1 else x for j, x in enumerate(line.split("\t"))) for line in p.read_text().splitlines()[1:]]
            if have != rows:
                bad.append(f"{name}/works.tsv")
        print("CHECK", "OK: every array and work list equals the files in" if not bad else f"FAILED {bad}", a.out)
        return 0 if not bad else 1
    for name, arrays in (("train", train), ("longval", longval)):
        (a.out / name).mkdir(parents=True, exist_ok=True)
        p = a.out / name / "episodes.npz"
        if p.exists():  # keep an identical file (a rewrite would change its md5 through the zip timestamps)
            z = np.load(p)
            if set(z.files) == set(arrays) and all(z[k].dtype == v.dtype and np.array_equal(z[k], v) for k, v in arrays.items()):
                continue
        np.savez(p, **arrays)
    (a.out / "train" / "works.tsv").write_text("row\topenalex_id\tcorpus_line\n" +
                                               "".join(f"{r}\t{p}\t{l}\n" for r, p, l in works))
    (a.out / "longval" / "works.tsv").write_text("row\topenalex_id\tcorpus_line\ttrain_row\n" +
                                                 "".join(f"{r}\t{p}\t{l}\t{t}\n" for r, p, l, t in long_works))
    print("wrote", a.out / "train", a.out / "longval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
