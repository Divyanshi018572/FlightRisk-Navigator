from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from path_utils import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR

st.set_page_config(page_title="Flight Risk Navigator", page_icon="✈", layout="wide")


@st.cache_data
def load_metadata() -> dict:
    meta_file = MODELS_DIR / "metadata.json"
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text(encoding="utf-8"))


@st.cache_resource
def load_preprocessor():
    file = MODELS_DIR / "preprocessor.pkl"
    return joblib.load(file) if file.exists() else None


@st.cache_data
def load_model_scores() -> dict:
    path = MODELS_DIR / "model_scores.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def available_models(scores: dict) -> list[str]:
    scored_models = list(scores.get("scores", {}).keys())
    if scored_models:
        return sorted([m for m in scored_models if (MODELS_DIR / f"{m}.pkl").exists()])
    return sorted([p.stem for p in MODELS_DIR.glob("*.pkl") if p.stem != "preprocessor"])


def model_reason(metrics: dict) -> str:
    opt_recall = float(metrics.get("opt_recall", 0.0))
    opt_precision = float(metrics.get("opt_precision", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))

    if opt_recall >= 0.70 and opt_precision >= 0.28:
        return "Strong delay catching power with acceptable alert quality."
    if roc_auc >= 0.74 and opt_recall >= 0.55:
        return "Stable ranking performance and good delayed-flight capture."
    if opt_precision >= 0.40 and opt_recall >= 0.40:
        return "Balanced tradeoff between missed delays and false alerts."
    if opt_precision >= 0.45:
        return "Conservative model with cleaner alerts when precision is critical."
    return "Useful baseline for comparison and fallback."


def rank_top_models_for_use_case(scores: dict, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for model_name, metrics in scores.get("scores", {}).items():
        opt_recall = float(metrics.get("opt_recall", metrics.get("recall", 0.0)))
        opt_precision = float(metrics.get("opt_precision", metrics.get("precision", 0.0)))
        opt_f2 = float(metrics.get("opt_f2", metrics.get("f2", 0.0)))
        roc_auc = float(metrics.get("roc_auc", 0.0))
        threshold = float(metrics.get("opt_threshold", 0.50))

        # Recall-first ranking for this use case (delay catching).
        use_case_score = (0.45 * opt_recall) + (0.25 * opt_f2) + (0.20 * opt_precision) + (0.10 * roc_auc)
        rows.append(
            {
                "model": model_name,
                "recommended_threshold": threshold,
                "opt_recall": opt_recall,
                "opt_precision": opt_precision,
                "opt_f2": opt_f2,
                "roc_auc": roc_auc,
                "use_case_score": use_case_score,
                "why_best": model_reason(metrics),
            }
        )

    if not rows:
        return pd.DataFrame()

    ranked = pd.DataFrame(rows).sort_values("use_case_score", ascending=False).head(top_n).reset_index(drop=True)
    ranked.index = ranked.index + 1
    return ranked


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "High Risk"
    if prob >= 0.4:
        return "Moderate Risk"
    return "Low Risk"


def get_model_threshold(scores: dict, model_name: str) -> float:
    by_map = scores.get("optimized_thresholds", {})
    if model_name in by_map:
        return float(by_map[model_name])
    per_model = scores.get("scores", {}).get(model_name, {})
    if "opt_threshold" in per_model:
        return float(per_model["opt_threshold"])
    return 0.5


@st.cache_data
def load_test_split() -> tuple[pd.DataFrame | None, pd.Series | None]:
    x_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_path = PROCESSED_DATA_DIR / "y_test.csv"
    if not x_path.exists() or not y_path.exists():
        return None, None

    x_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze("columns").astype(int)
    return x_test, y_test


@st.cache_data
def get_test_probabilities(model_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    x_test, y_test = load_test_split()
    if x_test is None or y_test is None:
        return None, None

    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        return None, None

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        return None, None

    proba = model.predict_proba(x_test)[:, 1]
    return proba, y_test.to_numpy()


def threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def top_factors(model, feature_vector: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        contributions = model.feature_importances_ * feature_vector.values[0]
    elif hasattr(model, "coef_"):
        contributions = model.coef_[0] * feature_vector.values[0]
    else:
        return pd.DataFrame(columns=["feature", "impact"])

    out = pd.DataFrame({"feature": feature_names, "impact": contributions})
    out["abs_impact"] = out["impact"].abs()
    return out.sort_values("abs_impact", ascending=False).head(8)


def show_project_overview(scores: dict, ranked_models: pd.DataFrame) -> None:
    st.title("Flight Risk Navigator")
    st.markdown(
        """
This app predicts whether a flight will arrive **15+ minutes late** using the 2015 US DOT/Kaggle dataset.

**Use case**
- Airlines: crew and gate planning
- Airports: congestion forecasting
- Passengers: proactive travel decisions

**Prediction window**
- Current setup is **pre-departure** by default (no leakage columns).
"""
    )
    st.subheader("Project Overview")
    st.markdown(
        """
- Dataset: US DOT/BTS 2015 flights (Kaggle), real-world records only
- Learning objective: classify delayed vs on-time flights
- Business objective: maximize delayed-flight catch rate while controlling false alerts
- Modeling strategy: multi-model benchmark + threshold optimization for operations use
"""
    )
    if not ranked_models.empty:
        top_model = ranked_models.iloc[0]
        st.info(
            "Use-case recommended model: "
            f"**{top_model['model']}** at threshold **{top_model['recommended_threshold']:.2f}** "
            f"(Recall {top_model['opt_recall']:.2f}, Precision {top_model['opt_precision']:.2f})."
        )


def show_evaluation_section(scores: dict) -> None:
    st.subheader("Model Evaluation")
    if not scores:
        st.info("No evaluation artifacts found yet. Run the pipeline to generate results.")
        return

    score_rows = []
    for model_name, metrics in scores.get("scores", {}).items():
        row = {"model": model_name, **metrics}
        score_rows.append(row)
    if score_rows:
        st.dataframe(pd.DataFrame(score_rows).sort_values("roc_auc", ascending=False, na_position="last"))

    cols = st.columns(3)
    if (OUTPUTS_DIR / "confusion_matrix.png").exists():
        cols[0].image(str(OUTPUTS_DIR / "confusion_matrix.png"), caption="Confusion Matrix")
    if (OUTPUTS_DIR / "roc_curve.png").exists():
        cols[1].image(str(OUTPUTS_DIR / "roc_curve.png"), caption="ROC Curves")
    if (OUTPUTS_DIR / "feature_importance.png").exists():
        cols[2].image(str(OUTPUTS_DIR / "feature_importance.png"), caption="Feature Importance")


def main() -> None:
    metadata = load_metadata()
    preprocessor = load_preprocessor()
    scores = load_model_scores()
    ranked_models = rank_top_models_for_use_case(scores, top_n=5)
    show_project_overview(scores, ranked_models)

    models = available_models(scores)
    if not models or preprocessor is None:
        st.warning("Model artifacts not found. Run pipeline scripts first.")
        show_evaluation_section(scores)
        return

    st.sidebar.title("Model Picker")
    st.sidebar.caption("Top 5 ranked for this use case (delay recall priority).")

    if not ranked_models.empty:
        sidebar_view = ranked_models[
            ["model", "recommended_threshold", "opt_recall", "opt_precision", "roc_auc", "why_best"]
        ].copy()
        sidebar_view["recommended_threshold"] = sidebar_view["recommended_threshold"].round(2)
        sidebar_view["opt_recall"] = sidebar_view["opt_recall"].round(3)
        sidebar_view["opt_precision"] = sidebar_view["opt_precision"].round(3)
        sidebar_view["roc_auc"] = sidebar_view["roc_auc"].round(3)
        st.sidebar.dataframe(sidebar_view, use_container_width=True)

        top_models = ranked_models["model"].tolist()
        default_model = scores.get("best_recall_model", top_models[0])
        default_index = top_models.index(default_model) if default_model in top_models else 0
        model_choice = st.sidebar.selectbox("Choose from Top 5 Models", top_models, index=default_index)
    else:
        model_choice = st.sidebar.selectbox("Choose model", models, index=0)

    st.subheader("Predict Delay")
    model = joblib.load(MODELS_DIR / f"{model_choice}.pkl")
    base_threshold = get_model_threshold(scores, model_choice)
    use_recommended_threshold = st.sidebar.checkbox("Use recommended threshold", value=True)
    threshold_default = float(round(base_threshold, 2)) if use_recommended_threshold else 0.50
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=threshold_default,
        step=0.01,
        help="Lower threshold increases delay recall; higher threshold reduces false alarms.",
    )
    st.caption(f"Model suggested threshold: {base_threshold:.2f}")

    defaults = metadata.get("default_values", {})
    c1, c2, c3 = st.columns(3)

    airline = c1.text_input("AIRLINE", value=str(defaults.get("AIRLINE", "AA")))
    origin = c2.text_input("ORIGIN_AIRPORT", value=str(defaults.get("ORIGIN_AIRPORT", "ATL")))
    destination = c3.text_input("DESTINATION_AIRPORT", value=str(defaults.get("DESTINATION_AIRPORT", "LAX")))

    c4, c5, c6 = st.columns(3)
    month = c4.number_input("MONTH", min_value=1, max_value=12, value=int(defaults.get("MONTH", 1)))
    day = c5.number_input("DAY", min_value=1, max_value=31, value=int(defaults.get("DAY", 1)))
    day_of_week = c6.number_input("DAY_OF_WEEK", min_value=1, max_value=7, value=int(defaults.get("DAY_OF_WEEK", 1)))

    c7, c8, c9 = st.columns(3)
    sched_dep = c7.number_input(
        "SCHEDULED_DEPARTURE (HHMM)",
        min_value=0,
        max_value=2359,
        value=int(defaults.get("SCHEDULED_DEPARTURE", 900)),
    )
    sched_time = c8.number_input("SCHEDULED_TIME", min_value=10.0, value=float(defaults.get("SCHEDULED_TIME", 120.0)))
    distance = c9.number_input("DISTANCE", min_value=10.0, value=float(defaults.get("DISTANCE", 500.0)))

    taxi_out = st.number_input("TAXI_OUT", min_value=0.0, value=float(defaults.get("TAXI_OUT", 15.0)))

    if metadata.get("include_departure_delay", False):
        departure_delay = st.number_input(
            "DEPARTURE_DELAY",
            value=float(defaults.get("DEPARTURE_DELAY", 0.0)),
            help="Included only if in-flight/at-gate prediction mode is used.",
        )
    else:
        departure_delay = 0.0

    if st.button("Predict", type="primary"):
        raw_input = {
            "AIRLINE": [airline],
            "ORIGIN_AIRPORT": [origin],
            "DESTINATION_AIRPORT": [destination],
            "MONTH": [month],
            "DAY": [day],
            "DAY_OF_WEEK": [day_of_week],
            "SCHEDULED_DEPARTURE": [sched_dep],
            "SCHEDULED_TIME": [sched_time],
            "DISTANCE": [distance],
            "TAXI_OUT": [taxi_out],
        }
        if metadata.get("include_departure_delay", False):
            raw_input["DEPARTURE_DELAY"] = [departure_delay]

        input_df = pd.DataFrame(raw_input)
        x = preprocessor.transform(input_df)
        prob = float(model.predict_proba(x)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.predict(x)[0])
        pred = int(prob >= threshold)

        st.metric("Delay Probability", f"{prob:.2%}", risk_label(prob))
        st.caption(f"Decision threshold for `{model_choice}`: {threshold:.2f}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.write(f"Predicted class: {'Delayed (>=15 min)' if pred == 1 else 'On-time (<15 min)'}")

        factors = top_factors(model, x, list(x.columns))
        if not factors.empty:
            st.subheader("Top Contributing Factors")
            st.bar_chart(factors.set_index("feature")["impact"])

    st.subheader("Threshold Analysis On Test Set")
    proba_test, y_test_np = get_test_probabilities(model_choice)
    if proba_test is None or y_test_np is None:
        st.info("No test split/probabilities found yet. Run preprocessing + training first.")
    else:
        diag = threshold_metrics(y_test_np, proba_test, threshold)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{diag['accuracy']:.2%}")
        c2.metric("Precision", f"{diag['precision']:.2%}")
        c3.metric("Recall", f"{diag['recall']:.2%}")
        c4.metric("F1", f"{diag['f1']:.2%}")

        cm_df = pd.DataFrame(
            [[diag["tn"], diag["fp"]], [diag["fn"], diag["tp"]]],
            index=["Actual On-time", "Actual Delayed"],
            columns=["Pred On-time", "Pred Delayed"],
        )
        st.dataframe(cm_df, use_container_width=True)

    show_evaluation_section(scores)


if __name__ == "__main__":
    main()

