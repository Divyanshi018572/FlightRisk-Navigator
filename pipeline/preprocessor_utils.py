from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FlightDelayPreprocessor:
    categorical_cols: List[str]
    numeric_cols: List[str]
    add_route: bool = False
    scaler: StandardScaler = field(default_factory=StandardScaler)
    freq_maps: Dict[str, Dict[str, float]] = field(default_factory=dict)
    medians: Dict[str, float] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["SCHEDULED_DEPARTURE"] = data["SCHEDULED_DEPARTURE"].fillna(0)
        data["SCHEDULED_DEPARTURE_HOUR"] = (data["SCHEDULED_DEPARTURE"] // 100).astype(int)
        data["IS_WEEKEND"] = data["DAY_OF_WEEK"].isin([6, 7]).astype(int)
        if self.add_route:
            data["ROUTE"] = (
                data["ORIGIN_AIRPORT"].astype(str) + "_" + data["DESTINATION_AIRPORT"].astype(str)
            )
            if "ROUTE" not in self.categorical_cols:
                self.categorical_cols = [*self.categorical_cols, "ROUTE"]
        return data

    def fit(self, df: pd.DataFrame) -> "FlightDelayPreprocessor":
        data = self._engineer_features(df)

        for col in self.numeric_cols:
            self.medians[col] = float(data[col].median())
            data[col] = data[col].fillna(self.medians[col])

        self.scaler.fit(data[self.numeric_cols])

        for col in self.categorical_cols:
            freq = data[col].astype(str).value_counts(normalize=True)
            self.freq_maps[col] = freq.to_dict()

        self.feature_names_ = [
            *self.numeric_cols,
            *[f"{col}_FREQ" for col in self.categorical_cols],
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._engineer_features(df)

        numeric_frame = pd.DataFrame(index=data.index)
        for col in self.numeric_cols:
            median = self.medians.get(col, 0.0)
            numeric_frame[col] = data[col].fillna(median)

        scaled_values = self.scaler.transform(numeric_frame[self.numeric_cols])
        transformed = pd.DataFrame(scaled_values, columns=self.numeric_cols, index=data.index)

        for col in self.categorical_cols:
            fmap = self.freq_maps.get(col, {})
            transformed[f"{col}_FREQ"] = data[col].astype(str).map(fmap).fillna(0.0)

        return transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
