#!/usr/bin/env python3
"""Fit the negative screen's latent-class model for one encoder (Section 4.1 of the paper), before training.

Fit sample: the in-batch (query, reviewer) pairs of 300 minibatches of 128 training episodes drawn like the training loop
draws them (seed 2026, not a training seed), with reviewers represented by their training support sets. Evidence per
pair: authorship Z, text T (largest frozen cosine to the support papers at least t_T) and citations G (largest Jaccard
similarity of reference lists at least t_G, or missing), see muster/screen.py. Thresholds: t_T and t_G are 95th
percentiles over the non-author pairs (t_G over the pairs where it is defined). Groups h: tertiles of the training
history lengths. A two-class latent-class model per group is fitted by maximum a posteriori EM (Dirichlet smoothing 1,
20 random starts), class 1 being the class where authorship is more frequent. Cross-fitting: the papers fall into two
folds, and the posteriors for one fold come from the model fitted on the other fold.

usage: python scripts/fit_screen.py --encoder sp2 [--out data/screen/model_sp2.json] [--device cuda] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muster.config import L_MAX_TRAIN, L_REC_TRAIN  # noqa: E402
from muster.data import load_train_pack, screen_files  # noqa: E402
from muster.screen import ReferenceIndex, author_row_sets, categories, fold_of, t_values  # noqa: E402
from muster.support import episode_supports  # noqa: E402

N_BATCH, BATCH, SEED, LAMBDA, STARTS = 300, 128, 2026, 1.0, 20


def em(counts, lam=LAMBDA, starts=STARTS, iters=5000, tol=1e-10, seed=0):
    """counts (2, 2, 3) over (Z, T, G) for one group -> the best penalized fit and the posterior p(E = 1 | Z, T, G)."""
    rng = np.random.default_rng(seed)
    best = None
    n = counts.sum()
    for _ in range(starts):
        pi = rng.dirichlet([1, 1])
        bz, bt, bg = rng.dirichlet([1, 1], 2), rng.dirichlet([1, 1], 2), rng.dirichlet([1, 1, 1], 2)
        prev = -np.inf
        for _ in range(iters):
            joint = pi[:, None, None, None] * bz[:, :, None, None] * bt[:, None, :, None] * bg[:, None, None, :]
            tot = joint.sum(0, keepdims=True)
            resp = joint / np.maximum(tot, 1e-300)
            ec = resp * counts[None]
            nv = ec.reshape(2, -1).sum(1)
            pi = (nv + lam) / (n + 2 * lam)
            bz = (ec.sum((2, 3)) + lam) / (nv[:, None] + 2 * lam)
            bt = (ec.sum((1, 3)) + lam) / (nv[:, None] + 2 * lam)
            bg = (ec.sum((1, 2)) + lam) / (nv[:, None] + 3 * lam)
            ll = float((counts * np.log(np.maximum(tot[0], 1e-300))).sum()) + lam * float(
                np.log(pi).sum() + np.log(bz).sum() + np.log(bt).sum() + np.log(bg).sum())
            if abs(ll - prev) < tol * max(1.0, abs(ll)):
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, pi, bz, bt, bg)
    ll, pi, bz, bt, bg = best
    if bz[1, 1] < bz[0, 1]:  # class 1 = the author-enriched class
        pi, bz, bt, bg = pi[::-1].copy(), bz[::-1].copy(), bt[::-1].copy(), bg[::-1].copy()
    joint = pi[:, None, None, None] * bz[:, :, None, None] * bt[:, None, :, None] * bg[:, None, None, :]
    return {"loglik": ll, "pi": pi.tolist(), "P_Z": bz.tolist(), "P_T": bt.tolist(), "P_G": bg.tolist()}, joint[1] / joint.sum(0)


def fit(encoder: str, device: torch.device):
    t0 = time.time()
    pack = load_train_pack(encoder)
    vecs = pack["vecs_train"].astype(np.float32)
    vg = torch.from_numpy(vecs).to(device)
    tq, th, tm, ta = pack["train_q_idx"], pack["train_h_idx"], pack["train_m"].astype(bool), pack["ep_author"]
    s_idx, s_m = episode_supports(vecs, th, tm, L_MAX_TRAIN, L_REC_TRAIN)
    n_hist = tm.sum(1)
    cuts = np.quantile(n_hist, [1 / 3, 2 / 3]).astype(int).tolist()
    h_of = np.digitize(n_hist, cuts, right=True).astype(np.int8)
    authors = author_row_sets(pack["seq_indptr"], pack["seq_idx"])
    refs = ReferenceIndex(screen_files(encoder)[1])
    perm = np.random.default_rng(SEED).permutation(len(tq))
    rec = {k: [] for k in ("Z", "vT", "vG", "qhas", "nuse", "nsup", "h", "fold")}
    for b in range(N_BATCH):
        idx = perm[b * BATCH:(b + 1) * BATCH]
        q = tq[idx]
        vq = vg[torch.from_numpy(q.astype(np.int64)).to(device)]
        vs = vg[torch.from_numpy(s_idx[idx].astype(np.int64)).to(device)]
        v_t = t_values(vq, vs, torch.from_numpy(s_m[idx]).to(device)).cpu().numpy()
        v_g, n_use, n_sup, q_has = refs.g_values(q, s_idx[idx], s_m[idx])
        Z = np.array([[int(qq) in authors[int(a)] for a in ta[idx]] for qq in q], dtype=np.int8)
        assert (np.diag(Z) == 1).all(), "an episode's query is not in its author's record"
        for k, v in (("Z", Z), ("vT", v_t), ("vG", v_g), ("qhas", np.broadcast_to(q_has[:, None], Z.shape)),
                     ("nuse", np.broadcast_to(n_use[None, :], Z.shape)), ("nsup", np.broadcast_to(n_sup[None, :], Z.shape)),
                     ("h", np.broadcast_to(h_of[idx][None, :], Z.shape)), ("fold", np.broadcast_to(fold_of(q)[:, None], Z.shape))):
            rec[k].append(np.array(v).reshape(-1))
    D = {k: np.concatenate(v) for k, v in rec.items()}
    non = D["Z"] == 0
    t_t = float(np.quantile(D["vT"][non], 0.95))
    vg_def = D["vG"][non & ~np.isnan(D["vG"])]
    t_g, g_rule = float(np.quantile(vg_def, 0.95)), "95th percentile"
    if t_g <= 0:
        t_g, g_rule = float(np.min(vg_def[vg_def > 0])), "any shared reference (95th percentile is 0)"
    T, G = categories(D["vT"], D["vG"], D["qhas"], D["nuse"], D["nsup"], t_t, t_g)
    counts = np.zeros((2, 3, 2, 2, 3))
    np.add.at(counts, (D["fold"], D["h"], D["Z"], T, G), 1)
    fits, post_for_fold = {}, np.zeros((2, 3, 2, 2, 3))
    for k in (0, 1):
        for h in range(3):
            f, post = em(counts[1 - k, h], seed=17 + 10 * k + h)  # fold k's posteriors from the model fitted on fold 1 - k
            fits[f"fold{k}_h{h}"] = f
            post_for_fold[k, h] = post
    return {"encoder": encoder, "t_T": t_t, "t_G": t_g, "t_G_rule": g_rule, "h_cuts": cuts, "lambda": LAMBDA,
            "starts": STARTS, "fit_sample": {"batches": N_BATCH, "batch": BATCH, "seed": SEED, "pairs": int(len(D["Z"])),
                                             "author_pairs": int(D["Z"].sum())},
            "posterior_for_fold": post_for_fold.tolist(), "fits": fits, "sec": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["sp2", "sci"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check", action="store_true", help="compare with the released model instead of writing")
    a = ap.parse_args()
    out = a.out or Path(screen_files(a.encoder)[0])
    model = fit(a.encoder, torch.device(a.device))
    if a.check:
        ref = json.loads(out.read_text())
        dp = float(np.abs(np.asarray(model["posterior_for_fold"]) - np.asarray(ref["posterior_for_fold"])).max())
        same = {k: model[k] == ref[k] for k in ("t_T", "t_G", "h_cuts")}
        same["fit_sample"] = all(model["fit_sample"][k] == ref["fit_sample"][k] for k in model["fit_sample"])
        print(json.dumps({"equal": same, "posterior_max_abs_diff": dp, "t_T": [model["t_T"], ref["t_T"]],
                          "t_G": [model["t_G"], ref["t_G"]]}))
        return 0 if all(same.values()) and dp == 0.0 else 1
    out.write_text(json.dumps(model, indent=1))
    print(f"wrote {out}: t_T {model['t_T']:.4f}, t_G {model['t_G']:.4f}, h cuts {model['h_cuts']} ({model['sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
