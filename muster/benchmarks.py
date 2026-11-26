"""The seven standard benchmarks, loaded from their released files in benchmarks/raw/.

Preference (pairwise comparisons of expertise):
  LR-PC   212 comparisons: one submission, two reviewers with graded expertise (LR-Bench, paper-centric)
  LR-RC   1,184 comparisons: one reviewer, two submissions (LR-Bench, reviewer-centric)
  CMU     1,653 comparisons from the self-reported expertise of 58 researchers on 477 papers (CMU Gold Standard),
          each scored with the 10 released profile samples d_20_1 ... d_20_10
Ranking (rank a pool of reviewers for each submission, soft and hard relevance labels):
  NIPS, SIGIR, SciRepEval, KDD

Every benchmark is a set of scored items (submission row, reviewer key) over its own paper rows, with the reviewers'
histories as row lists (newest first where the benchmark gives years). The paper rows index the frozen encodings in
benchmarks/encodings/<enc>/. Orders follow the evaluation code behind the paper's tables, so a score array computed
here lines up with the released score files: LR items sorted by (paper id, author id), CMU items in the order of the
evaluations file, ranking items sorted by (query id, reviewer id).
"""
from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "benchmarks" / "raw"
ENCODINGS = REPO / "benchmarks" / "encodings"
RANK_DATASETS = ("NIPS", "SIGIR", "SciRepEval", "KDD")
GOLD_SAMPLES = tuple(f"d_20_{k}" for k in range(1, 11))


@dataclass
class ItemSet:
    """Items to score: the submission row and the reviewer of each item, and each reviewer's history (paper rows)."""
    query_rows: np.ndarray
    reviewer_keys: List[str]
    histories: Dict[str, List[int]]


@dataclass
class PreferenceBenchmarks:
    lr_ids: List[str]  # rows of the LR-Bench encodings
    lr_items: ItemSet  # 897 (paper, author) items shared by LR-PC and LR-RC
    lr_pairs: List[Tuple[str, str]]
    gold_ids: List[str]  # rows of the CMU encodings
    gold_items: List[ItemSet]  # one per profile sample, 477 (paper, participant) items each
    gold_pairs: List[Tuple[str, str]]
    tracks: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # pc, rc, cmu: a, b, td, w


@dataclass
class RankingBenchmark:
    name: str
    paper_ids: List[str]  # rows of the encodings
    items: ItemSet
    pairs: List[Tuple[str, str]]  # (query id, reviewer id) of each item
    queries: Dict[str, list]  # soft and hard: [{"query_id": ..., "score": {reviewer: 0 or 1}}] in file order


def _json(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    with gzip.open(Path(str(path) + ".gz"), "rt") as f:
        return json.load(f)


def load_preference(raw: Path = RAW, encodings: Path = ENCODINGS) -> PreferenceBenchmarks:
    lr_ids = [str(x) for x in json.load(open(encodings / "ids" / "lr.json"))]
    gold_ids = [str(x) for x in json.load(open(encodings / "ids" / "cmu.json"))]
    lr_row = {pid: i for i, pid in enumerate(lr_ids)}
    gold_row = {pid: i for i, pid in enumerate(gold_ids)}

    # CMU Gold Standard: self-reported expertise (participants x up to 10 papers) and 10 profile samples
    gold_refs: Dict[str, Dict[str, float]] = {}
    with open(raw / "cmu_gold" / "data" / "evaluations.csv", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            papers: Dict[str, float] = {}
            for x in range(1, 11):
                p_val, e_val = row.get(f"Paper{x}", ""), row.get(f"Expertise{x}", "")
                if p_val and e_val and p_val in gold_row:
                    papers[p_val] = float(e_val)
            gold_refs[str(row["ParticipantID"])] = papers
    gold_samples: Dict[str, Dict[str, List[int]]] = {}
    for sample in GOLD_SAMPLES:
        arch_dir = raw / "cmu_gold" / "evaluation_datasets" / sample / "archives"
        prof: Dict[str, List[int]] = {}
        for pid in gold_refs:
            idxs: List[int] = []
            arch_file = arch_dir / f"~{pid}.jsonl"
            if arch_file.exists():
                with open(arch_file) as af:
                    for line in af:
                        if line.strip():
                            doc_id = str(json.loads(line)["id"])
                            if doc_id in gold_row:
                                idxs.append(gold_row[doc_id])
            prof[pid] = idxs
        gold_samples[sample] = prof

    # LR-Bench: paper-centric and reviewer-centric examples; an author's history is its "author#k" rows in order
    lr_reviewers: Dict[str, List[int]] = {}
    splits: Dict[str, List[dict]] = {}
    for fname, tag in (("evaluations_pc", "pc"), ("evaluations_rc", "rc")):
        examples = []
        for ex in json.load(open(raw / "lr_bench" / f"{fname}.json")):
            a, pos, neg = ex["anchor"], ex["positive"], ex["negative"]
            if tag == "pc":
                for side in (pos, neg):
                    aid = str(side["author_id"])
                    lr_reviewers[aid] = [lr_row[f"{aid}#{i}"] for i in range(len(side["papers"])) if f"{aid}#{i}" in lr_row]
                pairs = [(str(a["paper_id"]), str(side["author_id"]), float(side["score"])) for side in (pos, neg)]
            else:
                aid = str(a["author_id"])
                lr_reviewers[aid] = [lr_row[f"{aid}#{i}"] for i in range(len(a["papers"])) if f"{aid}#{i}" in lr_row]
                pairs = [(str(side["paper_id"]), aid, float(side["score"])) for side in (pos, neg)]
            examples.append({"pos": pairs[0], "neg": pairs[1]})
        splits[tag] = examples
    lr_pairs = sorted({(p, a) for ex in splits["pc"] + splits["rc"] for p, a, _ in (ex["pos"], ex["neg"])})
    lr_pos = {pair: i for i, pair in enumerate(lr_pairs)}

    tracks: Dict[str, Dict[str, np.ndarray]] = {}
    for tag in ("pc", "rc"):
        a_idx, b_idx, td = [], [], []
        for ex in splits[tag]:
            (pp, pa, ps), (np_, na, ns) = ex["pos"], ex["neg"]
            diff = float(ps - ns)
            if diff == 0.0:
                continue
            a_idx.append(lr_pos[(pp, pa)])
            b_idx.append(lr_pos[(np_, na)])
            td.append(diff)
        tracks[tag] = _track(a_idx, b_idx, td)
    gold_pairs = [(p, pid) for pid, papers in gold_refs.items() for p in papers]
    gold_pos = {pair: i for i, pair in enumerate(gold_pairs)}
    a_idx, b_idx, td = [], [], []
    for reviewer, refs in gold_refs.items():
        papers = list(refs.keys())
        if len(papers) < 2:
            continue
        for p1, p2 in combinations(papers, 2):
            diff = float(refs[p1] - refs[p2])
            if diff == 0.0:
                continue
            a_idx.append(gold_pos[(p1, reviewer)])
            b_idx.append(gold_pos[(p2, reviewer)])
            td.append(diff)
    tracks["cmu"] = _track(a_idx, b_idx, td)

    lr_items = ItemSet(query_rows=np.asarray([lr_row[p] for p, _ in lr_pairs], dtype=np.int64),
                       reviewer_keys=[a for _, a in lr_pairs],
                       histories={a: [int(x) for x in lr_reviewers.get(a, [])] for a in dict.fromkeys(a for _, a in lr_pairs)})
    gold_q = np.asarray([gold_row[p] for p, _ in gold_pairs], dtype=np.int64)
    gold_items = [ItemSet(query_rows=gold_q, reviewer_keys=[pid for _, pid in gold_pairs],
                          histories={pid: [int(x) for x in gold_samples[s].get(pid, [])]
                                     for pid in dict.fromkeys(pid for _, pid in gold_pairs)})
                  for s in GOLD_SAMPLES]
    return PreferenceBenchmarks(lr_ids=lr_ids, lr_items=lr_items, lr_pairs=lr_pairs, gold_ids=gold_ids,
                                gold_items=gold_items, gold_pairs=gold_pairs, tracks=tracks)


def _track(a_idx: Sequence[int], b_idx: Sequence[int], td: Sequence[float]) -> Dict[str, np.ndarray]:
    """One comparison track: item a should score above item b when td > 0 (below when td < 0), weight |td|."""
    td_arr = np.asarray(td, dtype=np.float32)
    return {"a": np.asarray(a_idx, dtype=np.int64), "b": np.asarray(b_idx, dtype=np.int64), "td": td_arr, "w": np.abs(td_arr)}


def load_ranking(name: str, raw: Path = RAW) -> RankingBenchmark:
    d = raw / name
    seen, paper_ids = set(), []
    for p in _json(d / f"{name}_papers_test.json"):
        pid = str(p["paper"])
        if pid not in seen:  # some papers files list a paper twice; the first entry is kept
            seen.add(pid)
            paper_ids.append(pid)
    pid_to_row = {pid: i for i, pid in enumerate(paper_ids)}
    years_path = d / f"{name}_paper_years.json"
    years = {str(k): float(v) for k, v in json.load(open(years_path)).items() if v is not None} if years_path.exists() else {}
    reviewers: Dict[str, List[str]] = {}
    for r in _json(d / f"{name}_reviewers_test.json"):
        pids = [str(p) for p in r["papers"] if str(p) in pid_to_row]
        # newest first by year (2005 where unknown), file order among equal years; repeated entries are kept
        reviewers[str(r["reviewer"])] = [p for _, p in sorted(enumerate(pids), key=lambda iv: (-years.get(iv[1], 2005.0), iv[0]))]
    queries = {split: _json(d / f"{name}_queries_test_{split}.json") for split in ("soft", "hard")}
    pairs = sorted({(str(q["query_id"]), str(r)) for split in queries.values() for q in split for r in q["score"]})
    items = ItemSet(query_rows=np.asarray([pid_to_row[q] for q, _ in pairs], dtype=np.int64),
                    reviewer_keys=[r for _, r in pairs],
                    histories={r: [pid_to_row[p] for p in reviewers.get(r, []) if p in pid_to_row]
                               for r in dict.fromkeys(r for _, r in pairs)})
    return RankingBenchmark(name=name, paper_ids=paper_ids, items=items, pairs=pairs, queries=queries)


def load_encodings(encoder: str, encodings: Path = ENCODINGS) -> Dict[str, np.ndarray]:
    """The frozen unit vectors behind the paper's tables: lr, cmu and one matrix per ranking benchmark."""
    d = encodings / encoder
    return {k: np.load(d / f"{k}.npy") for k in ("lr", "cmu") + RANK_DATASETS}
