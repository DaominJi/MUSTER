"""Tests of the panel-expansion code on a small synthetic instance: run with `python -m pytest tests`.

They check the exact assignment oracle against a from-scratch solver, that every exact-marginal greedy variant selects the
recruits of CertGreedy, and that the exact optimum lies between CertGreedy's gain and its a posteriori bound."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("ortools")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from expansion import flow as fl  # noqa: E402
from expansion.baselines import naive_greedy, stochastic_greedy, topk_retrieval  # noqa: E402
from expansion.certgreedy import VARIANTS, greedy  # noqa: E402
from expansion.exact import Problem, mip_optimum, sift  # noqa: E402
from expansion.scenarios import B, SCENARIOS, build, sub_instance, load  # noqa: E402
from make_synthetic_instance import make  # noqa: E402


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    return make(tmp_path_factory.mktemp("instance"), tag="toy", n_sub=60, n_rev=600, seed=1)


def panel_of(cfg, k):
    P = fl.Panel(cfg["A0"], cfg["d"], cfg["b0"], extra=k, bmax=int(max(cfg["b0"].max(), cfg["bC"].max())))
    P.solve()
    return P


def value_of(cfg, selected):
    cols = [cfg["A0"]] + [cfg["AC"][c][:, None].astype(np.int64) for c in selected]
    return fl.scratch_value(np.hstack(cols), cfg["d"], np.concatenate([cfg["b0"], cfg["bC"][list(selected)]]).astype(np.int64))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_assignment_oracle_is_exact(inst, scenario):
    cfg = build(inst, "toy", scenario, seed=0)
    P = panel_of(cfg, 3)
    assert P.value == fl.scratch_value(cfg["A0"], cfg["d"], cfg["b0"])  # the optimal assignment value
    assert P.check() >= 0 and P.dual_objective() == P.value  # valid potentials and strong duality
    rng = np.random.default_rng(0)
    for c in rng.choice(np.flatnonzero(fl.usable(cfg["AC"], cfg["bC"])), size=5, replace=False):
        gain = P.trial(cfg["AC"][c], cfg["bC"][c])  # InsertReviewer
        assert gain == value_of(cfg, [c]) - P.value
        assert P.check_trial() >= 0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_greedy_variants_select_the_same_recruits(inst, scenario):
    cfg = build(inst, "toy", scenario, seed=0)
    k = min(max(cfg["ks"]), 12)
    P0 = panel_of(cfg, k)
    runs = {name: greedy(copy.deepcopy(P0), cfg["AC"], cfg["bC"], k, n_dummy=cfg["n_dummy"], **kw) for name, kw in VARIANTS.items()}
    cert = runs["certgreedy"]
    assert len(cert.selected) > 0 and sum(cert.gains) == value_of(cfg, cert.selected) - P0.value
    for name, run in runs.items():
        assert run.selected == cert.selected and run.gains == cert.gains, name
    assert max(run.trials for run in runs.values()) == runs["incremental_flow"].trials  # pruning never adds evaluations
    for mode in ("potential", "tight"):
        assert greedy(copy.deepcopy(P0), cfg["AC"], cfg["bC"], k, mode=mode, n_dummy=cfg["n_dummy"]).selected == cert.selected
    naive, _ = naive_greedy(cfg["A0"], cfg["b0"], cfg["AC"], cfg["bC"], cfg["d"], 2)
    assert naive.selected == cert.selected[:2]


@pytest.mark.parametrize("scenario", ["topic_gap", "emergency"])
def test_exact_optimum_brackets_certgreedy(inst, scenario):
    cfg = build(inst, "toy", scenario, seed=0)
    A0, b0, AC, d, k = cfg["A0"], cfg["b0"], cfg["AC"], cfg["d"], 5
    bC = np.asarray(cfg["bC"], np.int64)
    P0 = panel_of(cfg, k)
    g = greedy(copy.deepcopy(P0), AC, bC, k, ub_js=(k,), n_dummy=cfg["n_dummy"])
    m = mip_optimum(A0, b0, AC.astype(np.int64), bC, d, k, time_limit=300.0, strong=True)
    assert m["status"] == 0
    opt = value_of(cfg, m["selected"]) - P0.value
    assert sum(g.gains) <= opt <= g.ub[k]
    assert sum(g.gains) >= (1 - 1 / np.e) * opt  # the greedy guarantee
    free = np.asarray(b0) >= A0.shape[0]  # placeholders
    pos = -np.ones(A0.shape[1], np.int64)
    pos[np.flatnonzero(~free)] = np.arange(int((~free).sum()))
    pairs = [(pos[w], q) for q, w in P0.assignment() if pos[w] >= 0]
    start = (np.array([p[0] for p in pairs], np.int64), np.array([p[1] for p in pairs], np.int64))
    r = sift(Problem(A0, b0, AC, bC, d, k), P0.value, g.selected[:k], start, 300.0)
    assert r["closed"] and r["value"] - P0.value == opt  # column generation finds the same optimum


def test_baselines_do_not_beat_the_optimum_bound(inst):
    cfg = build(inst, "toy", "topic_gap", seed=0)
    k = 10
    P0 = panel_of(cfg, k)
    g = greedy(copy.deepcopy(P0), cfg["AC"], cfg["bC"], k, ub_js=(k,), n_dummy=cfg["n_dummy"])
    retrieved = topk_retrieval(cfg["AC"], cfg["bC"], k, B)
    assert len(retrieved) == k and value_of(cfg, retrieved) - P0.value <= g.ub[k]
    s = stochastic_greedy(copy.deepcopy(P0), cfg["AC"], cfg["bC"], k, 0.1, np.random.default_rng(0))
    assert sum(s.gains) == value_of(cfg, s.selected) - P0.value <= g.ub[k]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_sub_instances_are_feasible(inst, scenario):
    from expansion.scenarios import gap_pool, gap_topics

    spec, A = load(inst, "toy")
    pool = gap_pool(spec, A, gap_topics(inst, "toy"))[1] if scenario == "topic_gap" else None
    A0, b0, AC, bC, d, ks, n_dummy = sub_instance(scenario, np.random.default_rng(0), spec, A, pool_gap=pool)
    assert fl.scratch_value(A0, d, b0) is not None and len(ks) > 0 and AC.shape[1] == A0.shape[0]
