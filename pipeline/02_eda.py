from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_utils import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

sns.set_theme(style="whitegrid")


def save_plot(fig: plt.Figure, filename: str) -> None:
    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame) -> None:
    by_airline = df.groupby("AIRLINE_NAME", as_index=False)["DELAYED"].mean().sort_values("DELAYED", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=by_airline, x="AIRLINE_NAME", y="DELAYED", ax=ax)
    ax.set_title("Delay Rate by Airline")
    ax.set_ylabel("Delay Rate")
    ax.set_xlabel("Airline")
    ax.tick_params(axis="x", rotation=70)
    save_plot(fig, "eda_delay_rate_by_airline.png")

    by_day = df.groupby("DAY_OF_WEEK", as_index=False)["DELAYED"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=by_day, x="DAY_OF_WEEK", y="DELAYED", marker="o", ax=ax)
    ax.set_title("Delay Rate by Day of Week")
    save_plot(fig, "eda_delay_rate_by_day.png")

    by_month = df.groupby("MONTH", as_index=False)["DELAYED"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=by_month, x="MONTH", y="DELAYED", marker="o", ax=ax)
    ax.set_title("Delay Rate by Month")
    save_plot(fig, "eda_delay_rate_by_month.png")

    temp = df.copy()
    temp["SCHEDULED_DEPARTURE_HOUR"] = (temp["SCHEDULED_DEPARTURE"].fillna(0) // 100).astype(int)
    by_hour = temp.groupby("SCHEDULED_DEPARTURE_HOUR", as_index=False)["DELAYED"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=by_hour, x="SCHEDULED_DEPARTURE_HOUR", y="DELAYED", marker="o", ax=ax)
    ax.set_title("Delay Rate by Scheduled Departure Hour")
    save_plot(fig, "eda_delay_rate_by_hour.png")

    top_origins = temp["ORIGIN_AIRPORT"].value_counts().head(20).index
    top_df = temp[temp["ORIGIN_AIRPORT"].isin(top_origins)]
    origin_delay = top_df.groupby("ORIGIN_AIRPORT", as_index=False)["DELAYED"].mean().sort_values("DELAYED", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=origin_delay, x="ORIGIN_AIRPORT", y="DELAYED", ax=ax)
    ax.set_title("Delay Rate by Origin Airport (Top 20 by Volume)")
    save_plot(fig, "eda_delay_rate_by_origin_top20.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["ARRIVAL_DELAY"].clip(lower=-30, upper=300), bins=80, kde=False, ax=ax)
    ax.set_title("Arrival Delay Distribution (Clipped)")
    ax.set_xlabel("Arrival Delay (minutes)")
    save_plot(fig, "eda_arrival_delay_distribution.png")

    sample = df[["DEPARTURE_DELAY", "ARRIVAL_DELAY"]].dropna().sample(min(20000, len(df)), random_state=42)
    corr = sample["DEPARTURE_DELAY"].corr(sample["ARRIVAL_DELAY"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=sample, x="DEPARTURE_DELAY", y="ARRIVAL_DELAY", alpha=0.2, s=12, ax=ax)
    ax.set_title(f"Departure vs Arrival Delay (corr={corr:.3f})")
    ax.set_xlim(-30, 300)
    ax.set_ylim(-30, 300)
    save_plot(fig, "eda_departure_vs_arrival_delay.png")

    insight_file = OUTPUTS_DIR / "eda_key_insights.txt"
    insights = [
        f"Overall delay rate: {df['DELAYED'].mean():.3f}",
        f"Top delay airline: {by_airline.iloc[0]['AIRLINE_NAME']} ({by_airline.iloc[0]['DELAYED']:.3f})",
        f"Highest delay day: {int(by_day.sort_values('DELAYED', ascending=False).iloc[0]['DAY_OF_WEEK'])}",
        f"Highest delay month: {int(by_month.sort_values('DELAYED', ascending=False).iloc[0]['MONTH'])}",
        f"Departure vs arrival delay correlation: {corr:.3f}",
    ]
    insight_file.write_text("\n".join(insights), encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    input_file = PROCESSED_DATA_DIR / "flights_cleaned.csv"
    if not input_file.exists():
        raise FileNotFoundError("Run pipeline/01_data_loading.py first.")

    df = pd.read_csv(input_file, low_memory=False)
    run_eda(df)
    print(f"EDA charts saved in {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
