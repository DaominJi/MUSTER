"""Paths and loaders of the training data (data/) and the trained heads (checkpoints/).

data/train/episodes.npz      shared by both encoders (rows index the 446,314 corpus works of data/train/works.tsv)
    train_q_idx (90000,)         the query paper of each training episode
    train_h_idx, train_m (90000, 48)  its history: the author's newest earlier papers, newest first, and their mask
    ep_author (90000,)           the episode's author (index into seq_*)
    seq_indptr, seq_idx          each training author's record (CSR), the authorship Z of the screen
    val_q_idx, val_h_idx, val_m  the 2,000 validation episodes (held-out authors), histories as above
    vco_indptr, vco_idx          co-authors among the validation authors (excluded from the hard pools)
data/train/vectors_<enc>.npy  (446314, 768) float32 unit vectors of the frozen encoder
data/train/mu_<enc>.npy       (1, 768) mean of the training vectors (the corpus mean mu_corp)
data/longval/episodes.npz    the same 2,000 validation authors with their full earlier records (rows index
                             data/longval/works.tsv): q, h, m (2000, 256), val_vco_indptr, val_vco_idx
data/longval/vectors_<enc>.npy  (48741, 768) float32
data/screen/model_<enc>.json  the fitted latent-class model of the negative screen
data/screen/refs_csr.npz      reference lists of the training works (rows as in vectors_<enc>.npy)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CHECKPOINTS = REPO / "checkpoints"


def load_train_pack(encoder: str, data_dir: Path = DATA) -> Dict[str, np.ndarray]:
    z = np.load(Path(data_dir) / "train" / "episodes.npz")
    pack = {k: z[k] for k in z.files}
    pack["vecs_train"] = np.load(Path(data_dir) / "train" / f"vectors_{encoder}.npy")
    pack["mu_train"] = np.load(Path(data_dir) / "train" / f"mu_{encoder}.npy")
    return pack


def load_long_validation(encoder: str, data_dir: Path = DATA) -> Dict[str, np.ndarray]:
    z = np.load(Path(data_dir) / "longval" / "episodes.npz")
    lv = {k: z[k] for k in z.files}
    lv["V"] = np.load(Path(data_dir) / "longval" / f"vectors_{encoder}.npy")
    return lv


def screen_files(encoder: str, data_dir: Path = DATA) -> Tuple[str, str]:
    return str(Path(data_dir) / "screen" / f"model_{encoder}.json"), str(Path(data_dir) / "screen" / "refs_csr.npz")


def checkpoint_path(encoder: str) -> Path:
    return CHECKPOINTS / f"muster_{encoder}.pt"
