"""Small, dependency-light checks and metric helpers for RecallRadar."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "report_count",
    "injury_share",
    "death_share",
    "report_growth",
    "manufacturer_count",
]


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, int, int]:
    """Return precision, positive hits, and evaluated rows for a ranked list."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if len(labels) == 0 or k <= 0:
        return 0.0, 0, 0
    take = min(k, len(labels))
    selected = np.argsort(-scores, kind="stable")[:take]
    hits = int(labels[selected].sum())
    return hits / take, hits, take


def assert_no_future_leakage(frame: pd.DataFrame) -> None:
    """Ensure event features end before the forward recall-label window begins."""
    required = {"feature_date_max", "label_window_start", "label_window_end"}
    missing = required.difference(frame.columns)
    if missing:
        raise AssertionError(f"missing leakage columns: {sorted(missing)}")
    feature_end = pd.to_datetime(frame["feature_date_max"])
    label_start = pd.to_datetime(frame["label_window_start"])
    label_end = pd.to_datetime(frame["label_window_end"])
    if bool((feature_end >= label_start).any()):
        raise AssertionError("a feature date reaches the label window")
    if bool((label_start > label_end).any()):
        raise AssertionError("a label window runs backwards")

