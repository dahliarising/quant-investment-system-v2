"""
SHAP 모델 설명력 모듈
- TreeExplainer 기반 Top-K 피처 기여도
- Permutation Importance (모든 모델 호환)
- Pipeline 자동 처리 (전처리 → 모델)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def shap_top_features(
    fitted_estimator,
    X_background: pd.DataFrame,
    X_single: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    SHAP TreeExplainer로 Top-K 기여도 분석

    Args:
        fitted_estimator: sklearn Pipeline 또는 트리 모델
        X_background: 배경 샘플 (50~200행 권장)
        X_single: 설명할 1행 DataFrame
        feature_names: 원래 피처명
        top_k: 상위 피처 수

    Returns:
        DataFrame[feature, shap_value, feature_value, direction]
    """
    import shap

    est = fitted_estimator
    Xb = X_background
    Xs = X_single

    # Pipeline이면 전처리 후 마지막 모델에 TreeExplainer
    if hasattr(fitted_estimator, "named_steps"):
        steps = list(fitted_estimator.named_steps.items())
        for name, step in steps[:-1]:
            Xb = step.transform(Xb)
            Xs = step.transform(Xs)
        est = steps[-1][1]
        Xb = pd.DataFrame(Xb)
        Xs = pd.DataFrame(Xs)

    explainer = shap.TreeExplainer(est)
    shap_vals = np.array(explainer.shap_values(Xs)).reshape(-1)
    feat_vals = np.array(Xs.iloc[0]).reshape(-1)

    fn = feature_names if len(feature_names) == len(shap_vals) else [f"f{i}" for i in range(len(shap_vals))]

    idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    return pd.DataFrame({
        "feature": [fn[i] for i in idx],
        "shap_value": shap_vals[idx],
        "feature_value": feat_vals[idx],
        "direction": ["+" if shap_vals[i] > 0 else "-" for i in idx],
        "abs_impact": np.abs(shap_vals[idx]),
    })


def permutation_importance_topk(
    model, X: pd.DataFrame, y: pd.Series,
    k: int = 10, n_repeats: int = 5, random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation Importance (모든 모델 호환)

    Returns:
        DataFrame[feature, importance_mean, importance_std]
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    imp = pd.DataFrame({
        "feature": X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return imp.nlargest(k, "importance_mean").reset_index(drop=True)


def shap_summary_data(
    fitted_estimator,
    X_data: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 200,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    SHAP Summary Plot용 데이터 반환

    Returns:
        (shap_values_array, X_display) for shap.summary_plot
    """
    import shap

    est = fitted_estimator
    Xd = X_data.head(max_samples).copy()

    if hasattr(fitted_estimator, "named_steps"):
        steps = list(fitted_estimator.named_steps.items())
        for name, step in steps[:-1]:
            Xd = step.transform(Xd)
        est = steps[-1][1]
        Xd = pd.DataFrame(Xd, columns=feature_names[:Xd.shape[1]] if len(feature_names) >= Xd.shape[1] else None)

    explainer = shap.TreeExplainer(est)
    shap_vals = explainer.shap_values(Xd)
    return np.array(shap_vals), Xd
