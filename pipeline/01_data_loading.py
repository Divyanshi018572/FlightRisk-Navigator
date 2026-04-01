from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs


REQUIRED_FILES = ["flights.csv", "airlines.csv", "airports.csv"]


def validate_files(raw_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (raw_dir / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required files in {raw_dir}: {missing_str}. "
            "Download from Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays"
        )


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flights = pd.read_csv(raw_dir / "flights.csv", low_memory=False)
    airlines = pd.read_csv(raw_dir / "airlines.csv", low_memory=False)
    airports = pd.read_csv(raw_dir / "airports.csv", low_memory=False)
    return flights, airlines, airports


def build_clean_dataset(
    flights: pd.DataFrame,
    airlines: pd.DataFrame,
    airports: pd.DataFrame,
    sample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    df = flights.copy()

    df = df[df["CANCELLED"] == 0].copy()
    df = df[df["ARRIVAL_DELAY"].notna()].copy()
    df["DELAYED"] = (df["ARRIVAL_DELAY"] >= 15).astype(int)

    airlines_small = airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"})
    df = df.merge(airlines_small, on="AIRLINE", how="left")

    airport_cols = ["IATA_CODE", "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE"]
    airports_small = airports[airport_cols].rename(
        columns={
            "IATA_CODE": "ORIGIN_AIRPORT",
            "AIRPORT": "ORIGIN_AIRPORT_NAME",
            "CITY": "ORIGIN_CITY",
            "STATE": "ORIGIN_STATE",
            "LATITUDE": "ORIGIN_LAT",
            "LONGITUDE": "ORIGIN_LON",
        }
    )
    df = df.merge(airports_small, on="ORIGIN_AIRPORT", how="left")

    airports_dest = airports[airport_cols].rename(
        columns={
            "IATA_CODE": "DESTINATION_AIRPORT",
            "AIRPORT": "DEST_AIRPORT_NAME",
            "CITY": "DEST_CITY",
            "STATE": "DEST_STATE",
            "LATITUDE": "DEST_LAT",
            "LONGITUDE": "DEST_LON",
        }
    )
    df = df.merge(airports_dest, on="DESTINATION_AIRPORT", how="left")

    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=random_state,
            stratify=df["DELAYED"],
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and clean flight delay dataset.")
    parser.add_argument("--sample-size", type=int, default=500000, help="Sample size for development.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-full-data", action="store_true", help="Disable sampling.")
    args = parser.parse_args()

    ensure_project_dirs()
    validate_files(RAW_DATA_DIR)

    flights, airlines, airports = load_raw_data(RAW_DATA_DIR)
    sample_size = None if args.use_full_data else args.sample_size
    cleaned = build_clean_dataset(flights, airlines, airports, sample_size, args.random_state)

    output_file = PROCESSED_DATA_DIR / "flights_cleaned.csv"
    cleaned.to_csv(output_file, index=False)

    print(f"Saved cleaned data to {output_file}")
    print(f"Rows: {len(cleaned):,} | Columns: {cleaned.shape[1]}")
    print("Delay rate:", round(cleaned["DELAYED"].mean() * 100, 2), "%")


if __name__ == "__main__":
    main()
