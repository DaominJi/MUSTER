"""Test configuration: PyTorch and Numba can each load their own OpenMP runtime, and two runtimes in one process may
crash. The tests therefore run Numba's parallel loops on its built-in workqueue threading layer."""

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
