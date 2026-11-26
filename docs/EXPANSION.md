# Panel Expansion

Code for the second part of the paper: budgeted reviewer panel expansion. Given an incumbent panel, external candidates
and a budget `k`, it selects up to `k` candidates that maximize the value of the optimal assignment of the enlarged
panel.

## Code Map

| Paper | Code |
|---|---|
| Optimal assignment of a panel (min-cost flow) | `expansion/flow.py`: `Panel.solve` |
| `InsertReviewer`: exact marginal gain of one candidate | `expansion/flow.py`: `Panel.trial`, `_insert` |
| Shadow prices and dual upper bounds | `expansion/flow.py`: `Panel.pi`, `_dual_bounds`; `expansion/certgreedy.py`: `tight_pi`, `priced_pi` |
| `CertGreedy`: certified panel expansion | `expansion/certgreedy.py`: `greedy` |
| Lazy Greedy, Incremental-Flow Greedy, CertGreedy without lazy evaluation | `expansion/certgreedy.py`: `greedy` with `lazy` and `dual` (see `VARIANTS`) |
| Naive Greedy, Stochastic Greedy, Top-k retrieval | `expansion/baselines.py` |
| Exact optimum (HiGHS) | `expansion/exact.py`: `mip_optimum`, `sift` |
| Initialization, topic gap and emergency scenarios | `expansion/scenarios.py`: `build`, `sub_instance` |

Affinities are stored as integers in micro-units (`round(a * 1e6)`), so assignment values, marginal gains and bounds
are exact integers. Every greedy variant breaks ties by the smallest candidate index, so the exact-marginal variants
return the same recruits and differ only in how many marginal gains they evaluate.

## Instance Format

A venue instance is three files in one directory, named by a tag such as `vldb2025`:

| File | Content |
|---|---|
| `<tag>_A.npy` | `(reviewers, submissions)` int32 affinities in micro-units. `-1` marks an ineligible pair (conflict of interest). |
| `<tag>_spec.npz` | `inc_pool`: one bool per reviewer, true for the venue's earlier authors, from whom incumbent panels are sampled. |
| `topic_<tag>.npz` | `sub` and `past`: unit vectors of the submissions and of the venue's earlier papers, used to find the fastest-growing topics. Needed for the topic-gap scenario only. |

In the paper's experiments the reviewers of a venue are the researchers of its domain in the expert repository, the
submissions are the venue's 2025 papers, and the affinities are the scores of the default MUSTER model. These instance
files are large binary arrays and are not tracked by Git. To run the code without them, write a synthetic instance in
the same format:

```bash
python scripts/make_synthetic_instance.py runs/toy_instance --tag toy
```

An optional `gap_topics_<tag>.npz` stores the topic clustering of a venue. When it is present, `scenarios.gap_topics`
returns it, so every machine builds identical topic-gap instances.

## Scenarios

Every submission needs `D = 3` reviews and every reviewer takes at most `B = 4` submissions (`expansion/scenarios.py`).

| Scenario | Incumbent panel | Candidates | Budgets |
|---|---|---|---|
| `initialization` | none: zero-affinity placeholder reviewers keep the assignment feasible | every reviewer | `ceil(1.1 T / B)` with `T` the number of required reviews |
| `topic_gap` | sampled from the incumbent pool after leaving out the half best suited to the two fastest-growing topics, with 10% spare capacity | every other reviewer | 10, 20, 50, 100, 200 |
| `emergency` | the remaining members of a full panel after 5% of the assigned members withdraw, with their spare capacity and placeholders | every reviewer outside the full panel | 5, 10, 20, 50 |

`--seed` selects the sampled panel of the topic-gap and emergency scenarios.

## Running

```bash
# CertGreedy, its variants and the baselines: gain, exact evaluations and seconds at every budget
NUMBA_NUM_THREADS=1 python scripts/expand.py runs/toy_instance toy topic_gap --seed 0

# add Stochastic Greedy, and keep a JSON record of the run
NUMBA_NUM_THREADS=1 python scripts/expand.py runs/toy_instance toy emergency \
    --methods retrieval,dual_only,lazy,incremental,naive,stochastic --out runs/expansion

# exact optimum per budget (column generation over candidates and edges, or --full-mip on small instances)
python scripts/expand_exact.py runs/toy_instance toy emergency --limit 3600

# CertGreedy against the exact optimum on sub-instances with 100 submissions and 500 candidates
python scripts/expand_subinstances.py runs/toy_instance toy topic_gap --reps 10
```

`scripts/expand.py` prints one row per budget and method. Gains are exact: the value of the reoptimized assignment of
the enlarged panel minus the incumbent value, in units of affinity.

## Timing Protocol

- Selection times are measured on one thread (`NUMBA_NUM_THREADS=1`), after an untimed call that compiles the code.
- Incremental-Flow Greedy and Naive Greedy evaluate every remaining candidate in every round. On large instances the
  script reports their cost as the exact number of evaluations times the time of one evaluation. For Incremental-Flow
  Greedy this is one `InsertReviewer` call, sampled on 100 candidates at five stages of the greedy run and interpolated
  in between. For Naive Greedy it is one from-scratch OR-Tools solve, the mean of the incumbent panel and of a sample
  at the final panel. Each of the two runs for real when this estimate is below `--real-unpruned` seconds, and such
  rows carry no `estimated` mark.
- `--max-seconds` ends a greedy variant after the round in which its clock passes the limit.

## Checks in the Code

- The incumbent assignment is compared with a from-scratch OR-Tools min-cost flow, its potentials must leave no
  negative reduced cost, and its dual objective must equal its value.
- Every certified bound is asserted to be at least the exact marginal gain it bounds.
- Every exact-marginal variant is asserted to select the recruits of CertGreedy.
- A proven optimum is asserted to lie between CertGreedy's gain and its a posteriori upper bound, and the value of every
  selection returned by the MIP is recomputed exactly.

`tests/test_expansion.py` runs these checks on a synthetic instance.
