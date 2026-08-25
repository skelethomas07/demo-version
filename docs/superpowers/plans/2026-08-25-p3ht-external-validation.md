# P3HT External Validation Implementation Plan

> **For Codex:** Execute this plan task by task. Use test-driven development and fresh verification before reporting results.

**Goal:** Retrain the retention-time model with paper-level separation, quantify the contribution of physics-informed diffusivity features, and report honest external error for target-aligned post-May-2026 P3HT cases.

**Architecture:** Add a standalone validation module that reuses the repository cleaning and model-building utilities without overwriting the deployed model. Candidate model choices are selected only by grouped cross-validation on the historical P3HT dataset. The chosen model is then evaluated on every eligible external P3HT row, while a nearest-domain subset is reported separately using a similarity rule that does not use retention time.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, joblib, unittest

---

## Task 1: Freeze eligibility and metric behavior in tests

**Files:**
- Create: `tests/test_p3ht_validation.py`
- Create: `p3ht_validation.py`

**Step 1: Write failing tests**

Cover these behaviors:

- external eligibility requires strict target alignment, P3HT channel, and publication date on or after 2026-06-01
- similarity uses input conditions only and must not inspect `Tau_ms`
- percentage error and paper-level summaries are calculated from actual predictions
- physics-feature ablation removes the diffusivity-derived feature family while leaving other fields intact
- grouped folds never put the same `Paper_ID` in both training and validation

**Step 2: Run tests and confirm failure**

Run: `python -m unittest tests.test_p3ht_validation -v`

Expected: import or assertion failures because the implementation does not exist.

**Step 3: Implement minimal functions**

Implement only enough to pass the tests:

- `filter_external_p3ht`
- `select_similarity_inputs`
- `calculate_similarity_score`
- `summarize_external_predictions`
- `drop_physics_derived_features`
- `make_group_splits`

**Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_p3ht_validation -v`

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

## Task 2: Implement grouped model selection and physics ablation

**Files:**
- Modify: `tests/test_p3ht_validation.py`
- Modify: `p3ht_validation.py`

**Step 1: Write failing tests**

Test that:

- training input is limited to valid historical P3HT rows
- the bottom-one-percent target rule is fitted on training rows only
- grouped cross-validation returns fold-level and aggregate metrics
- a candidate is selected by grouped validation MAE in log space, never by external error
- the physics-on and physics-off variants use identical folds

**Step 2: Run tests and confirm failure**

Run: `python -m unittest tests.test_p3ht_validation -v`

Expected: failures for missing training and selection functions.

**Step 3: Implement grouped training**

Add:

- historical workbook loader
- P3HT cleaning and feature construction
- `GroupKFold` evaluation by `Paper_ID`
- compact ExtraTrees candidate grid
- log-space MAE, RMSE, median factor error, and MAPE reporting
- physics-on versus physics-off ablation

The external paper targets must not be passed into candidate selection.

**Step 4: Run focused and full tests**

Run both unittest commands from Task 1 and confirm all pass.

## Task 3: Build auditable P3HT external dataset and run evaluation

**Files:**
- Modify: `external_validation_data.csv`
- Create: `validation_results/p3ht_grouped_cv.csv`
- Create: `validation_results/p3ht_external_predictions.csv`
- Create: `validation_results/p3ht_external_summary.json`
- Create: `validation_results/p3ht_external_report.md`

**Step 1: Confirm source fields**

For every eligible external row, retain DOI, publication date, exact reported retention constant, material, ion, process fields, and source notes. Do not infer unavailable fabrication fields from retention time.

**Step 2: Run the grouped validation command**

Run:

`python p3ht_validation.py --historical-data 'data/통합 data ver.5.xlsx' --external-data external_validation_data.csv --out validation_results`

Expected outputs:

- grouped cross-validation metrics for physics-on and physics-off variants
- chosen candidate based only on historical grouped validation
- predictions for all eligible external P3HT rows
- a separately labelled nearest-domain subset based only on process and material similarity

**Step 3: Inspect actual results**

Check:

- row count and paper count
- no duplicated `validation_id`
- no overlap of `Paper_ID` across grouped folds
- all predicted and actual retention times are finite and positive
- reported metrics can be recalculated from the prediction CSV

## Task 4: Add reproducibility documentation

**Files:**
- Modify: `EXTERNAL_VALIDATION_README.md`

Document:

- historical source and P3HT restriction
- paper-level grouping
- physics-derived feature family and ablation
- external inclusion rule
- why the nearest-domain result is not the overall external result
- exact reproduction command

## Task 5: Final verification

**Step 1: Run all tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

**Step 2: Re-run the complete evaluation**

Run the command from Task 3 from a clean output state.

**Step 3: Recalculate headline metrics independently**

Read `p3ht_external_predictions.csv` and independently recompute MAPE, median factor error, and the nearest-domain subset result.

**Step 4: Review working tree**

Run: `git status --short`

Confirm only planned files changed. Do not push because the connected repository is read-only for writes.

**Step 5: Ask before self-introduction wording**

Present the verified actual metrics and the exact proposed sentence. Do not insert or delete wording from the user-approved self-introduction until the user approves it.
