"""Pytest contract for both learner and reference implementations."""
import importlib
import os
from pathlib import Path
import numpy as np
import nbformat
import pytest

module = importlib.import_module(os.getenv("XGB_TEXTBOOK_TARGET", "exercises.student"))


@pytest.mark.parametrize(("y", "p", "expected"), [
    ([1, 2, 3], [1, 2, 3], 0.0),
    ([1, 2], [3, 4], 4.0),
    ([-1, 1], [1, -1], 4.0),
])
def test_mse(y, p, expected):
    assert module.mse(y, p) == pytest.approx(expected)


def test_mse_gradient():
    y = np.array([1., -2., 0.5])
    p = np.array([0., 1., 1.5])
    got = module.mse_gradient(y, p)
    np.testing.assert_allclose(got, 2 * (p-y)/len(y), rtol=1e-12)
    # finite-difference consistency
    eps = 1e-6
    for k in range(len(y)):
        dp = np.zeros(len(y))
        dp[k] = eps
        fd = (module.mse(y, p + dp) - module.mse(y, p - dp)) / (2*eps)
        assert got[k] == pytest.approx(fd, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize(("x", "y", "threshold", "gain"), [
    ([0, 1, 2, 3], [0, 0, 10, 10], 1.5, 100.0),
    ([1, 1, 2, 2], [0, 0, 2, 2], 1.5, 4.0),
    ([0, 2, 3, 5], [1, 1, 1, 1], None, 0.0),
])
def test_best_split(x, y, threshold, gain):
    t, g = module.best_split_1d(np.array(x), np.array(y))
    if threshold is None:
        assert t is None
    else:
        assert t == pytest.approx(threshold)
    assert g == pytest.approx(gain, abs=1e-9)


def test_best_split_leaf_constraint():
    t, g = module.best_split_1d([0, 1, 2, 3], [0, 0, 10, 10], min_samples_leaf=3)
    assert t is None and g == pytest.approx(0)


def test_stump():
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 1, 3, 3])
    stump = module.fit_stump(x, y)
    assert set(stump) == {"threshold", "left_value", "right_value"}
    assert stump["threshold"] == pytest.approx(1.5)
    np.testing.assert_allclose(module.predict_stump(x, stump), y)


def test_stump_unsplittable():
    stump = module.fit_stump([1, 1, 1], [1, 2, 3])
    assert stump["threshold"] is None
    np.testing.assert_allclose(module.predict_stump([0, 1, 99], stump), [2, 2, 2])


def test_bootstrap():
    a = module.bootstrap_indices(12, np.random.default_rng(99))
    b = module.bootstrap_indices(12, np.random.default_rng(99))
    assert isinstance(a, np.ndarray)
    assert a.shape == (12,) and np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a < 12))
    np.testing.assert_array_equal(a, b)
    # A sufficiently large sample should contain duplicate indices, as in sampling with replacement.
    large = module.bootstrap_indices(200, np.random.default_rng(33))
    assert len(np.unique(large)) < len(large)


def test_boosting():
    x = np.linspace(0, 1, 40)
    y = np.where(x < .5, -2.0, 3.0)
    model = module.fit_gradient_boosting_1d(x, y, n_estimators=20, learning_rate=.2)
    assert set(model) == {"init", "learning_rate", "stumps"}
    assert len(model["stumps"]) == 20
    p = module.predict_gradient_boosting_1d(x, model)
    assert p.shape == x.shape
    assert np.mean((p-y)**2) < .01


def test_boosting_zero_estimators():
    x = [0, 1, 2]
    y = [1, 4, 7]
    model = module.fit_gradient_boosting_1d(x, y, n_estimators=0)
    np.testing.assert_allclose(module.predict_gradient_boosting_1d(x, model), [4, 4, 4])


@pytest.mark.parametrize(("G", "H", "lam", "expected"), [
    (4.0, 2.0, 2.0, -1.0),
    (-6.0, 1.0, 2.0, 2.0),
    (0.0, 1.0, 0.0, 0.0),
])
def test_optimal_leaf_weight(G, H, lam, expected):
    assert module.optimal_leaf_weight(G, H, lam) == pytest.approx(expected)


def test_split_gain():
    # GL=-4, GR=4, HL=HR=2, lambda=2 -> 0.5*(16/4+16/4-0)=4
    assert module.split_gain(-4, 2, 4, 2, 2) == pytest.approx(4.0)
    assert module.split_gain(-4, 2, 4, 2, 2, gamma=5) == pytest.approx(-1.0)


def test_logistic_grad_hess():
    logits = np.array([-2.0, 0.0, 2.0])
    y = np.array([0.0, 1.0, 1.0])
    g, h = module.logistic_grad_hess(logits, y)
    p = 1/(1+np.exp(-logits))
    np.testing.assert_allclose(g, p-y, atol=1e-12)
    np.testing.assert_allclose(h, p*(1-p), atol=1e-12)
    assert np.all(h > 0)


def test_logistic_grad_hess_numerical_stability():
    g, h = module.logistic_grad_hess([-1000., 1000.], [0., 1.])
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(h))


def test_notebooks_valid():
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
    paths = sorted(notebook_dir.glob("*.ipynb"))
    assert len(paths) >= 7
    for path in paths:
        nbformat.validate(nbformat.read(path, as_version=4))

