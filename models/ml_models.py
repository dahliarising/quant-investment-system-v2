"""
ML 모델 학습 파이프라인
- Ridge, Random Forest, XGBoost, LightGBM 지원
- 전처리 자동화 (수치형/범주형 분리)
- Date-wise Cross-Validation (daily RankIC 기반)
- LightGBM Early Stopping
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainedRegressor:
    model_name: str
    pipe: Pipeline
    feature_names: List[str]


# ---------------------
# Model Builders
# ---------------------
def _try_build_xgb(random_state: int = 42) -> BaseEstimator:
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError("xgboost 미설치. pip install xgboost") from e
    return XGBRegressor(
        n_estimators=1200, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=random_state, n_jobs=4,
        objective="reg:squarederror",
    )


def _try_build_lgbm(random_state: int = 42) -> BaseEstimator:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise ImportError("lightgbm 미설치. pip install lightgbm") from e
    return LGBMRegressor(
        n_estimators=5000, learning_rate=0.02, num_leaves=31,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=5.0,
        random_state=random_state, n_jobs=-1,
        objective="regression", verbosity=-1,
    )


def build_model(model_name: str, random_state: int = 42) -> BaseEstimator:
    name = (model_name or "").lower().strip()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name in ("rf", "random_forest", "randomforest"):
        return RandomForestRegressor(
            n_estimators=800, random_state=random_state,
            n_jobs=-1, max_depth=None, min_samples_leaf=2,
        )
    if name in ("xgb", "xgboost"):
        return _try_build_xgb(random_state)
    if name in ("lgbm", "lightgbm"):
        return _try_build_lgbm(random_state)
    raise ValueError(f"Unknown model: '{model_name}'. Use: ridge, rf, xgb, lgbm")


# ---------------------
# Preprocessing
# ---------------------
def _build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return preprocess, num_cols, cat_cols


# ---------------------
# Train / Predict
# ---------------------
def train_regressor(
    X: pd.DataFrame, y: pd.Series,
    model_name: str = "ridge", random_state: int = 42,
) -> TrainedRegressor:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    X2, y2 = X.align(y, join="inner", axis=0)
    y2 = pd.to_numeric(y2, errors="coerce")
    m = y2.notna()
    X2, y2 = X2.loc[m], y2.loc[m]

    preprocess, _, _ = _build_preprocess(X2)
    reg = build_model(model_name, random_state=random_state)
    pipe = Pipeline([("preprocess", preprocess), ("reg", reg)])
    pipe.fit(X2, y2)

    try:
        feat_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        feat_names = list(X2.columns)
    return TrainedRegressor(model_name=model_name, pipe=pipe, feature_names=feat_names)


def predict_regressor(tr: TrainedRegressor, X: pd.DataFrame) -> pd.Series:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    preds = tr.pipe.predict(X)
    return pd.Series(preds, index=X.index, name="pred")


# ---------------------
# LightGBM Early Stopping
# ---------------------
def _fit_predict_lgbm_early_stopping(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    *, random_state: int = 42, early_stopping_rounds: int = 80,
    val_ratio: float = 0.2, min_val_dates: int = 30,
) -> np.ndarray:
    if not isinstance(X_train.index, pd.MultiIndex):
        raise ValueError("X_train must be MultiIndex (date, ticker)")

    train_dates = pd.DatetimeIndex(sorted(pd.unique(X_train.index.get_level_values(0))))
    n_dates = len(train_dates)
    n_val = max(int(n_dates * val_ratio), min_val_dates)
    n_val = min(n_val, max(n_dates - 1, 1))

    val_dates = train_dates[-n_val:]
    tr_dates = train_dates[:-n_val]
    if len(tr_dates) < 10:
        tr_dates = train_dates
        val_dates = train_dates[:0]

    X_tr = X_train.loc[(tr_dates, slice(None)), :].copy()
    y_tr = y_train.loc[X_tr.index].copy()
    X_val, y_val = None, None
    if len(val_dates) > 0:
        X_val = X_train.loc[(val_dates, slice(None)), :].copy()
        y_val = y_train.loc[X_val.index].copy()

    preprocess, _, _ = _build_preprocess(X_tr)
    Xtr_enc = preprocess.fit_transform(X_tr)
    Xte_enc = preprocess.transform(X_test)
    reg = _try_build_lgbm(random_state=random_state)

    fit_kwargs: Dict[str, Any] = {}
    if X_val is not None and y_val is not None and len(X_val) > 0:
        Xval_enc = preprocess.transform(X_val)
        try:
            import lightgbm as lgb
            fit_kwargs = {
                "eval_set": [(Xval_enc, y_val.values)],
                "eval_metric": "l2",
                "callbacks": [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
            }
        except Exception:
            fit_kwargs = {}

    reg.fit(Xtr_enc, y_tr.values, **fit_kwargs)
    return np.asarray(reg.predict(Xte_enc), dtype=float)


# ---------------------
# Evaluation Metrics
# ---------------------
def information_coefficient(y_true: pd.Series, y_pred: pd.Series, method: str = "spearman") -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    df = pd.DataFrame({"y": yt, "p": yp}).dropna()
    if df.empty:
        return float("nan")
    return float(df["y"].corr(df["p"], method=method))


def rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    return information_coefficient(y_true, y_pred, method="spearman")


def hit_rate(y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.0) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    df = pd.DataFrame({"y": yt, "p": yp}).dropna()
    if df.empty:
        return float("nan")
    return float(((df["y"] > threshold) == (df["p"] > threshold)).mean())


def daily_cross_sectional_rankic(
    pred_panel: pd.Series, true_panel: pd.Series, min_names: int = 5
) -> pd.Series:
    df = pd.DataFrame({"pred": pred_panel, "true": true_panel}).dropna()
    if not isinstance(df.index, pd.MultiIndex):
        return pd.Series(dtype=float)
    dates = df.index.get_level_values(0)
    results = {}
    for d in sorted(dates.unique()):
        sub = df.loc[d]
        if len(sub) < min_names:
            continue
        results[d] = float(sub["pred"].corr(sub["true"], method="spearman"))
    return pd.Series(results, name="rankic")


# ---------------------
# Date-wise CV with Daily RankIC
# ---------------------
def _date_folds(unique_dates: pd.DatetimeIndex, n_splits: int):
    d = pd.DatetimeIndex(sorted(unique_dates))
    n = len(d)
    if n_splits < 2 or n < (n_splits + 2):
        raise ValueError("Not enough dates for CV")
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[:n % n_splits] += 1
    indices = np.arange(n)
    cur = 0
    folds = []
    for i in range(n_splits):
        start, stop = cur, cur + fold_sizes[i]
        test_idx = indices[start:stop]
        train_idx = indices[:start]
        cur = stop
        if len(train_idx) == 0:
            continue
        folds.append((d[train_idx], d[test_idx]))
    return folds


def cross_validate_models_daily_rankic(
    X: pd.DataFrame, y: pd.Series, *,
    model_names: Iterable[str], n_splits: int = 5,
    random_state: int = 42, min_train_dates: int = 252, min_cs_names: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    # Ensure MultiIndex
    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("X must have MultiIndex (date, ticker)")

    X2, y2 = X.align(y, join="inner", axis=0)
    y2 = pd.to_numeric(y2, errors="coerce")
    m = y2.notna()
    X2, y2 = X2.loc[m], y2.loc[m]
    if len(X2) == 0:
        return pd.DataFrame(), {}

    dates = pd.DatetimeIndex(sorted(pd.unique(X2.index.get_level_values(0))))
    folds = _date_folds(dates, n_splits)
    daily_rankic_map: Dict[str, pd.Series] = {}
    rows: List[Dict[str, Any]] = []

    for mn in model_names:
        try:
            build_model(mn, random_state)
        except Exception as e:
            rows.append({"model": mn, "ok": False, "note": str(e),
                         "n_oof": 0, "rmse": np.nan, "mae": np.nan,
                         "rank_ic": np.nan, "hit_rate": np.nan,
                         "daily_rankic_mean": np.nan, "daily_rankic_std": np.nan})
            continue

        oof = pd.Series(index=X2.index, dtype=float)
        for train_dates, test_dates in folds:
            if len(train_dates) < min_train_dates:
                continue
            tr_idx = X2.index.get_level_values(0).isin(train_dates)
            te_idx = X2.index.get_level_values(0).isin(test_dates)
            X_tr, y_tr = X2.loc[tr_idx], y2.loc[tr_idx]
            X_te = X2.loc[te_idx]
            if len(X_te) == 0:
                continue

            if mn == "lgbm":
                try:
                    pred = _fit_predict_lgbm_early_stopping(X_tr, y_tr, X_te, random_state=random_state)
                    oof.loc[X_te.index] = pred
                except Exception:
                    continue
            else:
                preprocess, _, _ = _build_preprocess(X_tr)
                reg = build_model(mn, random_state)
                pipe = Pipeline([("preprocess", preprocess), ("reg", reg)])
                try:
                    pipe.fit(X_tr, y_tr)
                    oof.loc[X_te.index] = pipe.predict(X_te)
                except Exception:
                    continue

        valid = oof.notna() & y2.notna()
        n_oof = int(valid.sum())
        if n_oof < 100:
            rows.append({"model": mn, "ok": False, "note": "Too few OOF",
                         "n_oof": n_oof, "rmse": np.nan, "mae": np.nan,
                         "rank_ic": np.nan, "hit_rate": np.nan,
                         "daily_rankic_mean": np.nan, "daily_rankic_std": np.nan})
            continue

        y_true = y2.loc[valid]
        y_pred = oof.loc[valid]
        rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_val = float(mean_absolute_error(y_true, y_pred))
        ric = rank_ic(y_true, y_pred)
        hr = hit_rate(y_true, y_pred)

        daily = daily_cross_sectional_rankic(y_pred, y_true, min_names=min_cs_names)
        daily_rankic_map[mn] = daily
        d_mean = float(daily.dropna().mean()) if len(daily.dropna()) else np.nan
        d_std = float(daily.dropna().std()) if len(daily.dropna()) else np.nan

        rows.append({
            "model": mn, "ok": True, "note": "",
            "n_oof": n_oof, "rmse": rmse_val, "mae": mae_val,
            "rank_ic": ric, "hit_rate": hr,
            "daily_rankic_mean": d_mean, "daily_rankic_std": d_std,
        })

    perf = pd.DataFrame(rows)
    if not perf.empty:
        perf = perf.sort_values(["ok", "daily_rankic_mean"], ascending=[False, False]).reset_index(drop=True)
    return perf, daily_rankic_map
