"""Unit and optional data-consistency tests: run with `python -m pytest tests`."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from muster.model import MusterHead, load_head  # noqa: E402

CORPUS = REPO / "data" / "corpus" / "corpus_elite_ccf_core_thu.jsonl.gz"
TRAIN_EPISODES = REPO / "data" / "train" / "episodes.npz"
REFS_RAW = REPO / "data" / "screen" / "references.jsonl.gz"
REFS_CSR = REPO / "data" / "screen" / "refs_csr.npz"


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)


def test_seeded_head_is_deterministic():
    torch.manual_seed(42)
    a = MusterHead().state_dict()
    torch.manual_seed(42)
    b = MusterHead().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert sum(v.numel() for k, v in a.items() if k != "mu_corp") == 44_891_732


def test_scoring_paths_agree_fresh_head():
    """score_all_pairs and score_items compute the same function when the support set covers the full history."""
    torch.manual_seed(0)
    head = MusterHead().eval()
    head.proto_fp32 = True
    g = torch.Generator().manual_seed(0)
    X = F.normalize(torch.randn(60, 768, generator=g), dim=-1).numpy()
    hist = {"r0": list(range(10, 40)), "r1": list(range(40, 45)), "r2": [45]}
    keys = [r for r in hist for _ in range(10)]
    q = np.tile(np.arange(10), 3)
    items = head.score_items(X, q, keys, hist, torch.device("cpu"))
    Xt = torch.from_numpy(X)
    for i, r in enumerate(hist):
        H = Xt[hist[r]].unsqueeze(0)
        m = torch.ones(1, len(hist[r]), dtype=torch.bool)
        with torch.no_grad():
            allp = head.score_all_pairs(Xt[:10], H, m, H, m)["scores"][:, 0].numpy()
        assert np.allclose(items[i * 10:(i + 1) * 10], allp, atol=2e-5), (r, np.abs(items[i * 10:(i + 1) * 10] - allp).max())


@pytest.mark.parametrize("enc", ["sp2", "sci"])
def test_scoring_paths_agree_checkpoint(enc):
    ckpt = REPO / "checkpoints" / f"muster_{enc}.pt"
    if not ckpt.exists():
        pytest.skip(f"{ckpt.name} not downloaded")
    head = load_head(ckpt)
    g = torch.Generator().manual_seed(0)
    X = F.normalize(torch.randn(60, 768, generator=g), dim=-1).numpy()
    hist = {"r0": list(range(10, 40)), "r1": list(range(40, 45)), "r2": [45]}
    keys = [r for r in hist for _ in range(10)]
    q = np.tile(np.arange(10), 3)
    items = head.score_items(X, q, keys, hist, torch.device("cpu"))
    Xt = torch.from_numpy(X)
    for i, r in enumerate(hist):
        H = Xt[hist[r]].unsqueeze(0)
        m = torch.ones(1, len(hist[r]), dtype=torch.bool)
        with torch.no_grad():
            allp = head.score_all_pairs(Xt[:10], H, m, H, m)["scores"][:, 0].numpy()
        assert np.allclose(items[i * 10:(i + 1) * 10], allp, atol=2e-5), (r, np.abs(items[i * 10:(i + 1) * 10] - allp).max())


@pytest.mark.skipif(not (CORPUS.exists() and TRAIN_EPISODES.exists()), reason="corpus archive not downloaded")
def test_training_data_rebuilds_from_the_corpus():
    r = run("scripts/build_training_data.py", "--check")
    assert r.returncode == 0 and "CHECK OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.skipif(not (REFS_RAW.exists() and REFS_CSR.exists()), reason="reference archive not downloaded")
def test_reference_index_rebuilds():
    r = run("scripts/build_references.py", "--check")
    assert r.returncode == 0 and "CHECK OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
