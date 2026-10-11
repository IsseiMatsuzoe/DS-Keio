"""Numerically checked reference solutions. Try the exercises first."""
import numpy as np


def _vector(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError("empty input")
    return a


def mse(y_true, y_pred):
    y, p = _vector(y_true), _vector(y_pred)
    if y.shape != p.shape:
        raise ValueError("shape mismatch")
    return float(np.mean((p - y) ** 2))


def mse_gradient(y_true, y_pred):
    y, p = _vector(y_true), _vector(y_pred)
    if y.shape != p.shape:
        raise ValueError("shape mismatch")
    return 2.0 * (p - y) / len(y)


def best_split_1d(x, y, min_samples_leaf=1):
    x, y = _vector(x), _vector(y)
    if x.shape != y.shape or min_samples_leaf < 1:
        raise ValueError("invalid input")
    parent_sse = float(np.sum((y - np.mean(y)) ** 2))
    unique_x = np.unique(x)
    if unique_x.size < 2:
        return None, 0.0
    candidates = (unique_x[:-1] + unique_x[1:]) / 2.0
    best_threshold, best_gain = None, 0.0
    for threshold in candidates:
        left = x <= threshold
        if left.sum() < min_samples_leaf or (~left).sum() < min_samples_leaf:
            continue
        yl, yr = y[left], y[~left]
        child_sse = np.sum((yl - yl.mean()) ** 2) + np.sum((yr - yr.mean()) ** 2)
        gain = parent_sse - float(child_sse)
        if gain > best_gain + 1e-12:
            best_threshold, best_gain = float(threshold), float(gain)
    return best_threshold, best_gain


def fit_stump(x, y, min_samples_leaf=1):
    x, y = _vector(x), _vector(y)
    threshold, _ = best_split_1d(x, y, min_samples_leaf)
    if threshold is None:
        avg = float(np.mean(y))
        return {"threshold": None, "left_value": avg, "right_value": avg}
    left = x <= threshold
    return {"threshold": threshold, "left_value": float(y[left].mean()),
            "right_value": float(y[~left].mean())}


def predict_stump(x, stump):
    x = np.asarray(x, dtype=float).reshape(-1)
    if stump["threshold"] is None:
        return np.full(x.shape, stump["left_value"], dtype=float)
    return np.where(x <= stump["threshold"], stump["left_value"], stump["right_value"]).astype(float)


def bootstrap_indices(n_samples, rng):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return rng.integers(0, n_samples, size=n_samples)


def fit_gradient_boosting_1d(x, y, n_estimators=10, learning_rate=0.1):
    x, y = _vector(x), _vector(y)
    if len(x) != len(y) or n_estimators < 0 or learning_rate <= 0:
        raise ValueError("invalid input")
    init = float(np.mean(y))
    pred = np.full_like(y, init)
    stumps = []
    for _ in range(n_estimators):
        residual = y - pred
        stump = fit_stump(x, residual)
        pred += learning_rate * predict_stump(x, stump)
        stumps.append(stump)
    return {"init": init, "learning_rate": float(learning_rate), "stumps": stumps}


def predict_gradient_boosting_1d(x, model):
    x = np.asarray(x, dtype=float).reshape(-1)
    pred = np.full_like(x, model["init"], dtype=float)
    for stump in model["stumps"]:
        pred += model["learning_rate"] * predict_stump(x, stump)
    return pred


def optimal_leaf_weight(G, H, reg_lambda):
    if H + reg_lambda <= 0:
        raise ValueError("H + reg_lambda must be positive")
    return -G / (H + reg_lambda)


def split_gain(GL, HL, GR, HR, reg_lambda, gamma=0.0):
    if min(HL + reg_lambda, HR + reg_lambda, HL + HR + reg_lambda) <= 0:
        raise ValueError("nonpositive denominator")
    parent_G, parent_H = GL + GR, HL + HR
    return (0.5 * (GL**2 / (HL + reg_lambda)
            + GR**2 / (HR + reg_lambda)
            - parent_G**2 / (parent_H + reg_lambda)) - gamma)


def logistic_grad_hess(logits, y):
    logits, y = _vector(logits), _vector(y)
    if logits.shape != y.shape:
        raise ValueError("shape mismatch")
    # stable sigmoid without exp overflow
    p = np.empty_like(logits)
    positive = logits >= 0
    p[positive] = 1 / (1 + np.exp(-logits[positive]))
    z = np.exp(logits[~positive])
    p[~positive] = z / (1 + z)
    return p - y, p * (1 - p)
