from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def _load_csv_array(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_processed() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = _load_csv_array(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = _load_csv_array(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = np.loadtxt(PROCESSED_DATA_DIR / "y_train.csv", delimiter=",", skiprows=1)
    y_test = np.loadtxt(PROCESSED_DATA_DIR / "y_test.csv", delimiter=",", skiprows=1)
    return X_train, X_test, y_train.ravel().astype(int), y_test.ravel().astype(int)


def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "f2": float(fbeta_score(y_true, preds, beta=2, zero_division=0)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    optimize_for: str = "f2",
    min_precision: float = 0.45,
) -> float:
    thresholds = np.linspace(0.10, 0.90, 81)
    best_threshold = 0.50
    best_score = -1.0
    fallback_threshold = 0.50
    fallback_score = -1.0

    for threshold in thresholds:
        m = metrics_from_proba(y_true, proba, threshold)
        score = m.get(optimize_for, m["f2"])

        if score > fallback_score:
            fallback_score = score
            fallback_threshold = float(threshold)

        if m["precision"] >= min_precision and score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold if best_score >= 0 else fallback_threshold


def evaluate_model_bundle(
    model_name: str,
    estimator,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    optimize_for: str,
    min_precision: float,
) -> dict:
    print(f"Training {model_name}...")
    model = clone(estimator)
    model.fit(X_fit, y_fit)

    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model {model_name} does not support predict_proba.")

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    opt_threshold = find_best_threshold(y_val, val_proba, optimize_for=optimize_for, min_precision=min_precision)

    default_metrics = metrics_from_proba(y_test, test_proba, threshold=0.50)
    tuned_metrics = metrics_from_proba(y_test, test_proba, threshold=opt_threshold)
    auc = float(roc_auc_score(y_test, test_proba))

    final_model = clone(estimator)
    final_model.fit(X_full, y_full)
    joblib.dump(final_model, MODELS_DIR / f"{model_name}.pkl")

    result = {
        "accuracy": default_metrics["accuracy"],
        "precision": default_metrics["precision"],
        "recall": default_metrics["recall"],
        "f1": default_metrics["f1"],
        "f2": default_metrics["f2"],
        "roc_auc": auc,
        "opt_threshold": float(opt_threshold),
        "opt_accuracy": tuned_metrics["accuracy"],
        "opt_precision": tuned_metrics["precision"],
        "opt_recall": tuned_metrics["recall"],
        "opt_f1": tuned_metrics["f1"],
        "opt_f2": tuned_metrics["f2"],
        "optimize_for": optimize_for,
    }
    print(model_name, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train flight delay models with recall-aware thresholds.")
    parser.add_argument("--skip-hyperparam-search", action="store_true", help="Skip RandomizedSearchCV stage.")
    parser.add_argument(
        "--optimize-for",
        choices=["f1", "f2", "recall"],
        default="f2",
        help="Metric used to pick validation threshold.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.45,
        help="Minimum precision constraint when selecting threshold.",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    required = [PROCESSED_DATA_DIR / "X_train.csv", PROCESSED_DATA_DIR / "X_test.csv"]
    if not all(p.exists() for p in required):
        raise FileNotFoundError("Run pipeline/03_preprocessing.py first.")

    X_train, X_test, y_train, y_test = load_processed()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    pos_count = int((y_fit == 1).sum())
    neg_count = int((y_fit == 0).sum())
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    models = {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=1),
        "decision_tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=25, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            max_depth=24,
            min_samples_split=4,
            class_weight={0: 1.0, 1: 2.2},
            n_jobs=1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=320,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=320,
            max_depth=-1,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.9,
            class_weight={0: 1.0, 1: 2.2},
            random_state=42,
            n_jobs=1,
        )

    model_scores = {}

    for name, estimator in models.items():
        model_scores[name] = evaluate_model_bundle(
            model_name=name,
            estimator=estimator,
            X_fit=X_fit,
            y_fit=y_fit,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_full=X_train,
            y_full=y_train,
            optimize_for=args.optimize_for,
            min_precision=args.min_precision,
        )

    if not args.skip_hyperparam_search:
        tuned_candidates = {}

        rf_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(class_weight={0: 1.0, 1: 2.2}, random_state=42, n_jobs=1),
            param_distributions={
                "n_estimators": [160, 220, 300],
                "max_depth": [12, 18, 24, 32, None],
                "min_samples_split": [2, 4, 6, 8],
                "min_samples_leaf": [1, 2, 5, 10],
            },
            n_iter=6,
            cv=3,
            scoring="roc_auc",
            n_jobs=1,
            random_state=42,
        )
        rf_search.fit(X_fit, y_fit)
        tuned_candidates["random_forest_tuned"] = rf_search.best_estimator_

        if XGBClassifier is not None:
            xgb_search = RandomizedSearchCV(
                estimator=XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=42, scale_pos_weight=scale_pos_weight),
                param_distributions={
                    "n_estimators": [200, 300, 400],
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.05, 0.08, 0.12],
                    "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.7, 0.85, 1.0],
                },
                n_iter=6,
                cv=3,
                scoring="roc_auc",
                n_jobs=1,
                random_state=42,
            )
            xgb_search.fit(X_fit, y_fit)
            tuned_candidates["xgboost_tuned"] = xgb_search.best_estimator_

        if LGBMClassifier is not None:
            lgbm_search = RandomizedSearchCV(
                estimator=LGBMClassifier(class_weight={0: 1.0, 1: 2.2}, random_state=42, n_jobs=1),
                param_distributions={
                    "n_estimators": [200, 300, 400],
                    "max_depth": [-1, 10, 20],
                    "learning_rate": [0.05, 0.08, 0.12],
                    "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.7, 0.85, 1.0],
                },
                n_iter=6,
                cv=3,
                scoring="roc_auc",
                n_jobs=1,
                random_state=42,
            )
            lgbm_search.fit(X_fit, y_fit)
            tuned_candidates["lightgbm_tuned"] = lgbm_search.best_estimator_

        for name, estimator in tuned_candidates.items():
            model_scores[name] = evaluate_model_bundle(
                model_name=name,
                estimator=estimator,
                X_fit=X_fit,
                y_fit=y_fit,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                X_full=X_train,
                y_full=y_train,
                optimize_for=args.optimize_for,
                min_precision=args.min_precision,
            )

    best_by_auc = max(model_scores, key=lambda m: model_scores[m].get("roc_auc", 0.0))
    best_by_opt_recall = max(model_scores, key=lambda m: model_scores[m].get("opt_recall", 0.0))

    summary = {
        "scores": model_scores,
        "optimized_thresholds": {name: data["opt_threshold"] for name, data in model_scores.items()},
        "best_model": best_by_auc,
        "best_metrics": model_scores[best_by_auc],
        "best_recall_model": best_by_opt_recall,
        "best_recall_metrics": model_scores[best_by_opt_recall],
        "threshold_selection": {
            "optimize_for": args.optimize_for,
            "min_precision": args.min_precision,
            "notes": "Optimized threshold selected on validation split and evaluated on test split.",
        },
    }

    (MODELS_DIR / "model_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Best ROC-AUC model:", best_by_auc)
    print(json.dumps(model_scores[best_by_auc], indent=2))
    print("Best recall model:", best_by_opt_recall)
    print(json.dumps(model_scores[best_by_opt_recall], indent=2))


if __name__ == "__main__":
    main()
