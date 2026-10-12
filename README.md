# RecallRadar

A model that flags which FDA medical-device product codes are most likely to start a Class I or II recall in the next 12 months, explains each flag, and tests whether it beats report volume alone.

Live demo: https://recallradar.mjeneidi.com

## The problem

Device makers and hospital systems watch MAUDE adverse-event reports for early signs of a recall. Report volume mostly follows how widely a device is used, so a useful signal has to be evaluated against the plain answer: rank the biggest report counts first.

RecallRadar turns that question into a time-ordered experiment for a regulated environment. It builds one row per product code per quarter, keeps the prediction path deterministic, and exposes structured outputs with validation and guardrails. There is no LLM summarization in the flag itself. The reason panel is a SHAP explanation from the trained model.

## The result

On the held-out 2023 to 2024 set of 400 code-quarter rows (187, or 47%, later started a Class I or II recall within 12 months), RecallRadar reached a PR-AUC of **0.959**, against **0.928** for the prior-recall baseline and **0.491** for raw report volume. The model's 1,000-resample bootstrap 95% interval was **0.934 to 0.977**. Exact values: 0.958812, 0.927835, 0.491496.

PR-AUC is the headline because the top-50 cutoff saturates on this split: the model and the prior-recall baseline both hit 50 of 50 (precision@50 of 1.0), while report volume hit 35 of 50. On this split, adverse-event patterns add ranking signal beyond recall history. This is a useful signal-detection demo, not a claim that the model should make a clinical or regulatory decision.

## How it works

1. The data pipeline reads openFDA MAUDE events and Class I or II recalls for 40 quarters from 2015 through 2024.
2. Each code-quarter row records report count, injury share, death share, report growth, manufacturer count, prior recall count, and time since the last recall.
3. A row is labeled 1 when a new recall starts for that product code in the following 12 months.
4. One LightGBM model trains through 2021, validates on 2022, and is tested only on 2023 to 2024. It combines adverse-event patterns with prior recall history, and SHAP turns each score into a reason.
5. The static dashboard ranks recorded runs, compares precision@10 with the volume baseline, and shows the held-out precision@50 and PR-AUC results against both baselines.

## Data and license

The event data comes from the [openFDA Device Adverse Event API](https://api.fda.gov/device/event.json) and the label data comes from the [openFDA Device Recall API](https://api.fda.gov/device/recall.json). Both are free public endpoints with openFDA's CC0 license terms. The build uses no API key. Raw API responses stay in the ignored local cache. The small feature table and the precomputed dashboard JSON are included for the static demo.

## Evaluation method

The split is strict in time: training ends in 2021, validation is 2022, and testing is 2023 to 2024. The headline is precision among the top 50 flagged code-quarter rows. Baselines rank by raw report count and by the prior year's recall count. The model receives the same prior recall count plus time since last recall, alongside the adverse-event features. PR-AUC is reported beside the headline, with a 1,000-resample bootstrap interval for the model. A leakage assertion and test cover ensure prior-recall dates never pass the feature date. This is a hypothesis-testing mindset around a simple operational question, not a causal claim. The project treats MAUDE as a noisy event signal, not as a clinical signal-processing system.

## How to run

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python pipeline.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory site
```

The exact test gate is `.venv/bin/python -m pytest -q`. The pipeline reads the verified endpoints, caches responses in `data/cache`, writes `data/processed/features.csv`, and refreshes `site/results.json`.

## Limitations

The dataset keeps the top 50 report-volume codes per quarter, so it is a portfolio demo rather than a complete device surveillance system. MAUDE reports are noisy, product-code definitions can change, and a recall label does not establish that the reports caused the recall. The model ties the prior-recall baseline on precision@50, while its PR-AUC is higher on this held-out set. The static site does not include production MLOps, model monitoring, a model registry, or an approval workflow for a regulated deployment. No LLM call was needed or used.

## Credits

Built by Mohammad Jeneidi with Python, pandas, LightGBM, SHAP, scikit-learn, and the Project Site Kit. Public data is from the U.S. Food and Drug Administration's openFDA program.
