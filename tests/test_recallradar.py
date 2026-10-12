import numpy as np
import pandas as pd
import pytest

from recallradar import assert_no_future_leakage, precision_at_k, prior_recall_features


def test_precision_at_k_ranks_highest_scores_first():
    precision, hits, evaluated = precision_at_k(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.6, 0.9]), 2)
    assert (precision, hits, evaluated) == (0.5, 1, 2)


def test_precision_at_k_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        precision_at_k(np.array([1]), np.array([0.2, 0.4]), 1)


def test_leakage_guard_accepts_forward_window():
    frame = pd.DataFrame(
        {
            "feature_date_max": ["2021-03-31"],
            "label_window_start": ["2021-04-01"],
            "label_window_end": ["2022-03-31"],
        }
    )
    assert_no_future_leakage(frame)


def test_leakage_guard_rejects_feature_inside_label_window():
    frame = pd.DataFrame(
        {
            "feature_date_max": ["2021-04-01"],
            "label_window_start": ["2021-04-01"],
            "label_window_end": ["2022-03-31"],
        }
    )
    with pytest.raises(AssertionError):
        assert_no_future_leakage(frame)


def test_prior_recall_features_ignore_recalls_after_feature_date():
    count, days, last_date = prior_recall_features(
        [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-01")],
        "2021-03-31",
        "2020-04-01",
    )
    assert (count, days, last_date) == (2, 30, "2021-03-01")


def test_leakage_guard_rejects_future_prior_recall_date():
    frame = pd.DataFrame(
        {
            "feature_date_max": ["2021-03-31"],
            "label_window_start": ["2021-04-01"],
            "label_window_end": ["2022-03-31"],
            "last_recall_date": ["2021-04-01"],
        }
    )
    with pytest.raises(AssertionError):
        assert_no_future_leakage(frame)
