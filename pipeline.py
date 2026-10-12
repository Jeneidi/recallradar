"""Build RecallRadar from the two verified openFDA endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.metrics import average_precision_score

from recallradar import FEATURE_COLUMNS, assert_no_future_leakage, precision_at_k, prior_recall_features


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
SITE = ROOT / "site"
EVENT_URL = "https://api.fda.gov/device/event.json"
RECALL_URL = "https://api.fda.gov/device/recall.json"
HEADERS = {"User-Agent": "RecallRadar/1.0 (portfolio research project)"}
YEARS = range(2015, 2025)
TOP_CODES_PER_QUARTER = 50


def _cache_path(url: str, params: dict[str, Any]) -> Path:
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    return CACHE / f"{key}.json"


def fetch_json(url: str, params: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url, params)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    for attempt in range(3):
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if response.status_code == 404:
            return {"results": []}
        if response.status_code == 200:
            payload = response.json()
            path.write_text(json.dumps(payload))
            return payload
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
            time.sleep(5 * (2**attempt))
            continue
        response.raise_for_status()
    raise RuntimeError("openFDA request failed after retries")


def quarter_rows() -> list[dict[str, str]]:
    rows = []
    for start in pd.date_range("2015-01-01", "2024-10-01", freq="QS"):
        end = start + pd.offsets.QuarterEnd(0)
        rows.append(
            {
                "quarter": f"{start.year} Q{((start.month - 1) // 3) + 1}",
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "api_start": start.strftime("%Y%m%d"),
                "api_end": end.strftime("%Y%m%d"),
            }
        )
    return rows


def distribution(period: dict[str, str], event_type: str | None, refresh: bool) -> dict[str, int]:
    search = f"date_received:[{period['api_start']} TO {period['api_end']}]"
    if event_type:
        search += f" AND event_type:{event_type}"
    payload = fetch_json(
        EVENT_URL,
        {"search": search, "count": "device.device_report_product_code.exact", "limit": 100},
        refresh,
    )
    return {str(item["term"]): int(item["count"]) for item in payload.get("results", []) if item.get("term")}


def manufacturer_count(args: tuple[dict[str, str], str, bool]) -> tuple[str, str, int]:
    period, code, refresh = args
    search = (
        f"date_received:[{period['api_start']} TO {period['api_end']}] "
        f"AND device.device_report_product_code:{code}"
    )
    payload = fetch_json(
        EVENT_URL,
        {"search": search, "count": "device.manufacturer_d_name.exact", "limit": 100},
        refresh,
    )
    count = len([item for item in payload.get("results", []) if item.get("term")])
    return period["quarter"], code, count


def fetch_recall_records(refresh: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for year in range(2015, 2026):
        params_base = {
            "search": (
                f"event_date_initiated:[{year}-01-01 TO {year}-12-31] "
                "AND openfda.device_class:[1 TO 2]"
            ),
            "limit": 100,
        }
        first = fetch_json(RECALL_URL, {**params_base, "skip": 0}, refresh)
        total = int(first.get("meta", {}).get("results", {}).get("total", 0))
        for skip in range(0, total, 100):
            payload = first if skip == 0 else fetch_json(RECALL_URL, {**params_base, "skip": skip}, refresh)
            records.extend(payload.get("results", []))
        print(f"Fetched Class I and II recalls for {year} ({total} records)", flush=True)
    return records


def build_features(refresh: bool = False) -> pd.DataFrame:
    periods = quarter_rows()
    distributions: dict[tuple[str, str], dict[str, int]] = {}
    for index, period in enumerate(periods, start=1):
        for kind, event_type in (("all", None), ("injury", "Injury"), ("death", "Death")):
            distributions[(period["quarter"], kind)] = distribution(period, event_type, refresh)
        print(f"Fetched event distributions for {period['quarter']} ({index}/{len(periods)})", flush=True)

    candidates: list[tuple[dict[str, str], str]] = []
    raw_rows: list[dict[str, Any]] = []
    previous_counts: dict[str, int] = {}
    for period in periods:
        all_counts = distributions[(period["quarter"], "all")]
        injury_counts = distributions[(period["quarter"], "injury")]
        death_counts = distributions[(period["quarter"], "death")]
        top_codes = sorted(all_counts, key=all_counts.get, reverse=True)[:TOP_CODES_PER_QUARTER]
        for code in top_codes:
            total = all_counts[code]
            raw_rows.append(
                {
                    "quarter": period["quarter"],
                    "quarter_start": period["start"],
                    "quarter_end": period["end"],
                    "code": code,
                    "report_count": total,
                    "injury_count": injury_counts.get(code, 0),
                    "death_count": death_counts.get(code, 0),
                    "injury_share": injury_counts.get(code, 0) / total,
                    "death_share": death_counts.get(code, 0) / total,
                    "report_growth": (total - previous_counts.get(code, 0)) / max(previous_counts.get(code, 0), 1),
                }
            )
            candidates.append((period, code))
            previous_counts[code] = total

    manufacturers: dict[tuple[str, str], int] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(manufacturer_count, (period, code, refresh)) for period, code in candidates]
        for index, future in enumerate(as_completed(futures), start=1):
            quarter, code, count = future.result()
            manufacturers[(quarter, code)] = count
            if index % 100 == 0 or index == len(futures):
                print(f"Fetched manufacturer counts ({index}/{len(futures)})", flush=True)

    recalls = fetch_recall_records(refresh)
    recall_dates: dict[str, list[pd.Timestamp]] = {}
    for record in recalls:
        code = record.get("product_code")
        date = record.get("event_date_initiated")
        if code and date:
            recall_dates.setdefault(str(code), []).append(pd.Timestamp(date))
    for dates in recall_dates.values():
        dates.sort()

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        end = pd.Timestamp(row["quarter_end"])
        start = pd.Timestamp(row["quarter_start"])
        dates = recall_dates.get(row["code"], [])
        label_start = end + pd.Timedelta(days=1)
        label_end = end + pd.DateOffset(months=12)
        prior_start = start - pd.DateOffset(months=12)
        forward = [date for date in dates if label_start <= date <= label_end]
        prior_count, time_since_last_recall_days, last_recall_date = prior_recall_features(dates, end, prior_start)
        rows.append(
            {
                **row,
                "manufacturer_count": manufacturers[(row["quarter"], row["code"])],
                "prior_recall_count": prior_count,
                "time_since_last_recall_days": time_since_last_recall_days,
                "last_recall_date": last_recall_date,
                "label": int(bool(forward)),
                "label_window_start": label_start.strftime("%Y-%m-%d"),
                "label_window_end": label_end.strftime("%Y-%m-%d"),
                "feature_date_max": row["quarter_end"],
            }
        )
    frame = pd.DataFrame(rows)
    assert_no_future_leakage(frame)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "features.csv", index=False)
    return frame


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, metric: str, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(1000):
        indices = rng.integers(0, len(y), len(y))
        sample_y = y[indices]
        sample_scores = scores[indices]
        if metric == "pr_auc":
            if len(np.unique(sample_y)) < 2:
                continue
            values.append(float(average_precision_score(sample_y, sample_scores)))
        else:
            values.append(float(precision_at_k(sample_y, sample_scores, 50)[0]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def shap_reason(values: np.ndarray, feature_names: list[str]) -> str:
    positive = [(abs(float(value)), index) for index, value in enumerate(values) if value > 0]
    index = max(positive, default=(0.0, int(np.argmax(np.abs(values)))))[1]
    phrases = {
        "report_count": "report volume pushed the score upward",
        "injury_share": "the injury share pushed the score upward",
        "death_share": "the death share pushed the score upward",
        "report_growth": "recent report growth pushed the score upward",
        "manufacturer_count": "the manufacturer count pushed the score upward",
        "prior_recall_count": "recent recall history pushed the score upward",
        "time_since_last_recall_days": "time since the last recall pushed the score upward",
    }
    return phrases[feature_names[index]]


def run_model(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    frame["year"] = pd.to_datetime(frame["quarter_start"]).dt.year
    train = frame[frame["year"] <= 2021]
    validation = frame[frame["year"] == 2022]
    test = frame[frame["year"] >= 2023]
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])
    frame["score"] = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]

    test_y = test["label"].to_numpy()
    test_score = frame.loc[test.index, "score"].to_numpy()
    volume_score = test["report_count"].to_numpy(dtype=float)
    prior_score = test["prior_recall_count"].to_numpy(dtype=float)
    model_top = precision_at_k(test_y, test_score, 50)
    volume_top = precision_at_k(test_y, volume_score, 50)
    prior_top = precision_at_k(test_y, prior_score, 50)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(frame[FEATURE_COLUMNS])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    demo_quarters: dict[str, list[dict[str, Any]]] = {}
    for quarter, group in frame[frame["year"] >= 2022].groupby("quarter", sort=False):
        selected = group.sort_values("score", ascending=False).head(10)
        items = []
        for index, row in selected.iterrows():
            items.append(
                {
                    "code": row["code"],
                    "score": round(float(row["score"]), 6),
                    "label": int(row["label"]),
                    "report_count": int(row["report_count"]),
                    "injury_share": round(float(row["injury_share"]), 6),
                    "death_share": round(float(row["death_share"]), 6),
                    "report_growth": round(float(row["report_growth"]), 6),
                    "manufacturer_count": int(row["manufacturer_count"]),
                    "reason": shap_reason(shap_values[index], FEATURE_COLUMNS),
                }
            )
        demo_quarters[quarter] = items

    backtest = []
    for quarter, group in frame[frame["year"] >= 2022].groupby("quarter", sort=False):
        y = group["label"].to_numpy()
        backtest.append(
            {
                "quarter": quarter,
                "model_precision_at_10": round(precision_at_k(y, group["score"].to_numpy(), 10)[0], 6),
                "volume_precision_at_10": round(precision_at_k(y, group["report_count"].to_numpy(), 10)[0], 6),
            }
        )

    result = {
        "project": "RecallRadar",
        "recorded_with": "Python, LightGBM and SHAP",
        "recorded_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "data": {
            "rows": int(len(frame)),
            "quarters": int(frame["quarter"].nunique()),
            "unique_codes": int(frame["code"].nunique()),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "test_positive_rows": int(test["label"].sum()),
            "top_codes_per_quarter": TOP_CODES_PER_QUARTER,
        },
        "metrics": {
            "headline": {
                "metric": "precision_at_50",
                "model": round(float(model_top[0]), 6),
                "model_hits": int(model_top[1]),
                "volume_baseline": round(float(volume_top[0]), 6),
                "volume_hits": int(volume_top[1]),
                "prior_recall_baseline": round(float(prior_top[0]), 6),
                "prior_recall_hits": int(prior_top[1]),
                "model_ci_95": [round(x, 6) for x in bootstrap_ci(test_y, test_score, "precision_at_50")],
            },
            "pr_auc": {
                "model": round(float(average_precision_score(test_y, test_score)), 6),
                "volume_baseline": round(float(average_precision_score(test_y, volume_score)), 6),
                "prior_recall_baseline": round(float(average_precision_score(test_y, prior_score)), 6),
                "model_ci_95": [round(x, 6) for x in bootstrap_ci(test_y, test_score, "pr_auc", 43)],
            },
        },
        "backtest": backtest,
        "demo_quarters": demo_quarters,
        "leakage_guard": "Every feature_date_max is before label_window_start for every row.",
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "results.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="ignore the local API response cache")
    args = parser.parse_args()
    frame = build_features(refresh=args.refresh)
    result = run_model(frame)
    headline = result["metrics"]["headline"]
    pr_auc = result["metrics"]["pr_auc"]
    print(f"Rows: {result['data']['rows']}; train: {result['data']['train_rows']}; test: {result['data']['test_rows']}")
    print(f"Leakage guard: {result['leakage_guard']}")
    print(
        "Precision@50: "
        f"model={headline['model']:.6f} ({headline['model_hits']}/50), "
        f"volume={headline['volume_baseline']:.6f} ({headline['volume_hits']}/50), "
        f"prior_recall={headline['prior_recall_baseline']:.6f} ({headline['prior_recall_hits']}/50), "
        f"95% CI={headline['model_ci_95']}"
    )
    print(
        "PR-AUC: "
        f"model={pr_auc['model']:.6f}, volume={pr_auc['volume_baseline']:.6f}, "
        f"prior_recall={pr_auc['prior_recall_baseline']:.6f}, 95% CI={pr_auc['model_ci_95']}"
    )


if __name__ == "__main__":
    main()
