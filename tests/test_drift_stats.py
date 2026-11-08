import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "monitoring"))

from drift_stats import psi  # noqa: E402


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(100, 10, 5000)
    assert psi(x, x.copy()) < 0.01


def test_psi_is_small_for_samples_from_the_same_distribution():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(100, 10, 5000), rng.normal(100, 10, 5000)) < 0.10


def test_psi_flags_a_large_shift():
    rng = np.random.default_rng(2)
    assert psi(rng.normal(100, 10, 5000), rng.normal(130, 10, 5000)) > 0.25