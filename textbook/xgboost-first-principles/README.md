# XGBoostを第一原理から理解する（日本語）

**対象:** Python / NumPy の基礎と微分を知っている人。量子・物理系の数式に慣れている人でも、機械学習の前提は飛ばしません。

## 学習ゴール
1. 二乗損失・勾配・バイアスと分散を説明できる
2. CART の分割と葉の最適値を NumPy で実装できる
3. Bagging と Random Forest の分散低減を説明できる
4. Gradient Boosting の残差学習を手で追える
5. XGBoost の2次近似から葉の重みと split gain を導出できる
6. 検証データを適切に分離し、XGBoost を実データに適用できる

## 目次（順番に実行）
| 章 | Notebook | 重要な数学 | 実装課題 |
|---|---|---|---|
| 0 | [損失と勾配](notebooks/00_loss_and_gradients.ipynb) | MSE / chain rule / finite difference | mse, mse_gradient |
| 1 | [決定木とCART](notebooks/01_decision_trees.ipynb) | 分割・SSE・葉の平均 | best_split_1d, fit_stump, predict_stump |
| 2 | [BaggingからRandom Forest](notebooks/02_bagging_random_forest.ipynb) | 分散・相関・bootstrap | bootstrap_indices |
| 3 | [Gradient Boosting](notebooks/03_gradient_boosting.ipynb) | 関数空間の勾配・残差 | fit/predict_gradient_boosting_1d |
| 4 | [XGBoostの数式](notebooks/04_xgboost_math.ipynb) | Taylor 2次近似・Hessian・正則化 | optimal_leaf_weight, split_gain, logistic_grad_hess |
| 5 | [実データでXGBoost](notebooks/05_xgboost_practice.ipynb) | ログ損失・検証・早期終了 | 考察・パラメータ比較 |
| 6 | [統合演習](notebooks/06_capstone.ipynb) | bias–variance / モデル選択 | ベースライン比較・解釈 |

概念：**単一の木 → 多数の木の平均（Bagging） → 誤差を順に補う木（Boosting） → 2次の損失情報＋正則化（XGBoost）**。

## セットアップ（Python 3.11 推奨）
この README があるディレクトリへ移動して実行:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

VS Code の Notebook / GitHub Codespaces でも利用できます。データセットは scikit-learn 同梱のものだけを使用します。外部の認証や Kaggle ダウンロードは不要です。

## 演習の提出・採点
各 Notebook は `exercises/student.py` の TODO を編集する形式です。**解答は `solutions/reference.py` に別置き**（見る前に自力で挑戦）。

```bash
python tools/grade.py                         # 自分の実装を採点
python tools/grade.py -k test_best_split      # 特定項目だけ採点
python tools/grade.py --reference             # 解答例自身の健全性検証
```

`pytest` の失敗内容で関数ごとの間違いを確認できます。GitHub Actions は教材が完成済みでなくても赤くならないよう **解答例のテスト**・Notebook 構造検証・**全7章のNotebook実行**を CI で実施します。自分の達成状況は手元で `python tools/grade.py` を実行してください。公開テスト通過は想定ケースの合格であり、実装の唯一性やあらゆる入力での正しさを証明しません。

## カリキュラムの方針
- まず 1 次元・回帰問題で数式とアルゴリズムを自力実装する。後半で二値分類へ移る。
- Notebook の図は Matplotlib で自動生成。セルを再実行すると結果を変更できる。
- `sklearn` では「正解を利用した確認」を行うが、最初の実装演習は NumPy のみ。
- XGBoost が最初から全探索しているわけではない：実際の木構築では近似／hist などの実装上の高速化がある。本教材は目的関数と分割原理に焦点を当てる。
- 分割戦略は手法のコアであって、全てのソフトウェア実装が教材の簡易 1D スタンプと同じ挙動をするとは限らない。

## 参考（公式）
- [XGBoost 論文 Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)
- [XGBoost Python sklearn estimator](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
- [scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

