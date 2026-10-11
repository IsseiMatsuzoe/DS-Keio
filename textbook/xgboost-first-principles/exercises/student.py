"""演習用。TODO を順番に埋め、python tools/grade.py で採点する。"""
import numpy as np


def mse(y_true, y_pred):
    """平均二乗誤差 (1/n) Σ(yhat-y)^2 を返す。"""
    raise NotImplementedError("TODO 0-1: mse")


def mse_gradient(y_true, y_pred):
    """予測値 y_pred に関する mse の勾配（各要素）を返す。"""
    raise NotImplementedError("TODO 0-2: mse_gradient")


def best_split_1d(x, y, min_samples_leaf=1):
    """1D 回帰木: SSE reduction が最大の (threshold, gain)。
    候補: 隣接する distinct な x の中点。両側の件数は min_samples_leaf 以上。
    gain > 0 がない場合は (None, 0.0)。同点の場合は最小の閾値。
    """
    raise NotImplementedError("TODO 1-1: best_split_1d")


def fit_stump(x, y, min_samples_leaf=1):
    """深さ1の回帰木を返す。
    dict keys: threshold, left_value, right_value
    分割できない時 threshold=None で両値は y の平均。
    """
    raise NotImplementedError("TODO 1-2: fit_stump")


def predict_stump(x, stump):
    """fit_stump の dict を用いて 1D ndarray の予測値を返す。"""
    raise NotImplementedError("TODO 1-3: predict_stump")


def bootstrap_indices(n_samples, rng):
    """np.random.Generator rng を使い復元抽出（長さ n_samples の index）。"""
    raise NotImplementedError("TODO 2-1: bootstrap_indices")


def fit_gradient_boosting_1d(x, y, n_estimators=10, learning_rate=0.1):
    """二乗損失の残差へ depth-1 回帰木を順次 fit。
    dict keys: init（y平均）, learning_rate, stumps（list）。
    """
    raise NotImplementedError("TODO 3-1: fit_gradient_boosting_1d")


def predict_gradient_boosting_1d(x, model):
    """init + learning_rate * Σ predict_stump を返す。"""
    raise NotImplementedError("TODO 3-2: predict_gradient_boosting_1d")


def optimal_leaf_weight(G, H, reg_lambda):
    """XGBoost: -G/(H + reg_lambda)。H+reg_lambda>0 と仮定。"""
    raise NotImplementedError("TODO 4-1: optimal_leaf_weight")


def split_gain(GL, HL, GR, HR, reg_lambda, gamma=0.0):
    """XGBoost split gain (splitを実行する前の値。負も許す)。"""
    raise NotImplementedError("TODO 4-2: split_gain")


def logistic_grad_hess(logits, y):
    """二値 cross-entropy の logit に対する勾配・2階微分を ndarray の組で返す。
    g = sigmoid(logit)-y, h = sigmoid(logit)*(1-sigmoid(logit))
    """
    raise NotImplementedError("TODO 4-3: logistic_grad_hess")
