from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()

    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("Run preprocessing and model training first.")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    model_scores_path = MODELS_DIR / "model_scores.json"
    if not model_scores_path.exists():
        raise FileNotFoundError("Run pipeline/04_model_training.py first.")

    model_summary = json.loads(model_scores_path.read_text(encoding="utf-8"))
    available_models = [
        m for m in model_summary.get("scores", {}).keys() if (MODELS_DIR / f"{m}.pkl").exists()
    ]

    results = []

    can_plot = True
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
    except Exception:
        can_plot = False

    if can_plot:
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for model_name in sorted(available_models):
        model = joblib.load(MODELS_DIR / f"{model_name}.pkl")
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if proba is not None else None,
        }
        results.append(row)

        if can_plot and proba is not None:
            RocCurveDisplay.from_predictions(y_test, proba, ax=ax_roc, name=model_name)

    if can_plot:
        ax_roc.set_title("ROC Curves - Flight Delay Models")
        fig_roc.savefig(OUTPUTS_DIR / "roc_curve.png", dpi=200, bbox_inches="tight")
        plt.close(fig_roc)

    best_model_name = model_summary["best_model"]
    best_model = joblib.load(MODELS_DIR / f"{best_model_name}.pkl")
    best_preds = best_model.predict(X_test)

    if can_plot:
        cm = confusion_matrix(y_test, best_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap="Blues", values_format="d")
        ax_cm.set_title(f"Confusion Matrix - {best_model_name}")
        fig_cm.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
        plt.close(fig_cm)

        if hasattr(best_model, "feature_importances_"):
            feat_names = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv", nrows=1).columns.tolist()
            importance = (
                pd.DataFrame({"feature": feat_names, "importance": best_model.feature_importances_})
                .sort_values("importance", ascending=False)
                .head(10)
            )
            fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
            ax_imp.barh(importance["feature"], importance["importance"])
            ax_imp.invert_yaxis()
            ax_imp.set_title(f"Top 10 Feature Importance - {best_model_name}")
            fig_imp.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
            plt.close(fig_imp)

    report = classification_report(y_test, best_preds)
    scores_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False, na_position="last")
    scores_df.to_csv(OUTPUTS_DIR / "model_evaluation_summary.csv", index=False)

    (OUTPUTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print("Saved evaluation outputs in outputs/")
    if not can_plot:
        print("Plot generation skipped due to local matplotlib/numpy compatibility issue.")
    print(scores_df.to_string(index=False))


if __name__ == "__main__":
    main()
