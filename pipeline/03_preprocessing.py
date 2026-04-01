from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs
from pipeline.preprocessor_utils import FlightDelayPreprocessor


LEAKAGE_COLUMNS = [
    "ARRIVAL_DELAY",
    "ARRIVAL_TIME",
    "ELAPSED_TIME",
    "AIR_TIME",
    "TAXI_IN",
    "WHEELS_ON",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "CANCELLATION_REASON",
]

DROP_COLUMNS = [
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "YEAR",
]

TARGET_COL = "DELAYED"


def build_feature_frame(df: pd.DataFrame, include_departure_delay: bool = False) -> tuple[pd.DataFrame, pd.Series, dict]:
    keep_df = df.copy()

    cols_to_drop = [c for c in [*LEAKAGE_COLUMNS, *DROP_COLUMNS] if c in keep_df.columns]
    keep_df = keep_df.drop(columns=cols_to_drop)

    if not include_departure_delay and "DEPARTURE_DELAY" in keep_df.columns:
        keep_df = keep_df.drop(columns=["DEPARTURE_DELAY"])

    if "CANCELLED" in keep_df.columns:
        keep_df = keep_df.drop(columns=["CANCELLED"])

    y = keep_df[TARGET_COL].astype(int)
    X = keep_df.drop(columns=[TARGET_COL])

    numeric_cols = [
        c
        for c in [
            "MONTH",
            "DAY",
            "DAY_OF_WEEK",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_TIME",
            "DISTANCE",
            "TAXI_OUT",
            "DEPARTURE_DELAY",
        ]
        if c in X.columns
    ]

    categorical_cols = [c for c in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"] if c in X.columns]

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "include_departure_delay": include_departure_delay,
    }
    return X, y, metadata


def main() -> None:
    ensure_project_dirs()
    input_file = PROCESSED_DATA_DIR / "flights_cleaned.csv"
    if not input_file.exists():
        raise FileNotFoundError("Run pipeline/01_data_loading.py first.")

    df = pd.read_csv(input_file, low_memory=False)

    include_departure_delay = False
    X, y, metadata = build_feature_frame(df, include_departure_delay=include_departure_delay)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = FlightDelayPreprocessor(
        categorical_cols=metadata["categorical_cols"],
        numeric_cols=metadata["numeric_cols"],
        add_route=False,
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_train_transformed.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test_transformed.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    (PROCESSED_DATA_DIR / "feature_names.txt").write_text("\n".join(preprocessor.feature_names_), encoding="utf-8")

    metadata["feature_names"] = preprocessor.feature_names_
    metadata["default_values"] = {
        "AIRLINE": str(X["AIRLINE"].mode().iloc[0]) if "AIRLINE" in X.columns else "",
        "ORIGIN_AIRPORT": str(X["ORIGIN_AIRPORT"].mode().iloc[0]) if "ORIGIN_AIRPORT" in X.columns else "",
        "DESTINATION_AIRPORT": str(X["DESTINATION_AIRPORT"].mode().iloc[0]) if "DESTINATION_AIRPORT" in X.columns else "",
        "MONTH": int(X["MONTH"].mode().iloc[0]) if "MONTH" in X.columns else 1,
        "DAY": int(X["DAY"].mode().iloc[0]) if "DAY" in X.columns else 1,
        "DAY_OF_WEEK": int(X["DAY_OF_WEEK"].mode().iloc[0]) if "DAY_OF_WEEK" in X.columns else 1,
        "SCHEDULED_DEPARTURE": int(X["SCHEDULED_DEPARTURE"].median()) if "SCHEDULED_DEPARTURE" in X.columns else 900,
        "SCHEDULED_TIME": float(X["SCHEDULED_TIME"].median()) if "SCHEDULED_TIME" in X.columns else 120.0,
        "DISTANCE": float(X["DISTANCE"].median()) if "DISTANCE" in X.columns else 500.0,
        "TAXI_OUT": float(X["TAXI_OUT"].median()) if "TAXI_OUT" in X.columns else 15.0,
        "DEPARTURE_DELAY": float(X["DEPARTURE_DELAY"].median()) if "DEPARTURE_DELAY" in X.columns else 0.0,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved transformed train/test data and preprocessor.")
    print(f"Train shape: {X_train_transformed.shape}, Test shape: {X_test_transformed.shape}")


if __name__ == "__main__":
    main()
