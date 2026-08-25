# 2026 external validation package

This package runs the repository's real `models/retention_model.joblib` against
publication-held-out retention constants. The model returns `log1p(Tau_ms)`, so
the code converts predictions with `expm1` before calculating errors.

## Reproduce

Run from the repository root:

```bash
python -m unittest tests/test_external_validation.py
python external_validation.py \
  --data external_validation_data.csv \
  --model models/retention_model.joblib \
  --schema models/input_schema.json \
  --output-dir validation_results
```

The workflow creates:

- `validation_results/external_validation_predictions.csv`: paper values,
  predictions, row-level errors, missing-input counts, and out-of-domain flags.
- `validation_results/external_validation_summary.json`: internal holdout and
  external metrics.
- `validation_results/external_validation_report.md`: readable result summary.

## Inclusion rule

The publication window is 2026-06-01 through 2026-08-25. The primary `strict`
set includes only a directly reported PSC/EPSC decay constant. For
multi-exponential fits, the slow component used by the original capstone data
rule (`tau2` or `tau3`) is selected. Retained-current ratios such as “24.5% at
1000 s” and lower bounds such as “retention >10000 s” are not treated as tau.

The one `supplementary` row uses a PPF-index decay constant and is reported only
as a sensitivity check because its target is related to, but not identical to,
PSC retention.

## Search and screening scope

The screening combined publisher pages and scholarly web searches for terms
including `ion-gated synaptic transistor`, `OECT synaptic retention`,
`electrolyte-gated neuromorphic transistor`, `ionic decay constant`, and
`retention tau`, restricted to the date window. `paper_screening_log.csv`
records included and excluded candidates.

“All papers” should therefore be read as all papers found within this documented
search scope as of 2026-08-25, not a guarantee of global bibliographic
completeness. Papers with figure-only values are not digitized into the strict
set because that would add unreported extraction error.

## Result and interpretation

The strict set contains 8 conditions from 3 independent papers. Despite the
repository's random holdout `R2_log` of 0.8898, the external result is weak:

- external `R2_log1p`: -3.1900
- external `RMSE_log1p`: 4.9220
- median multiplicative error: 150.331x
- within 3x: 25.0%
- MAPE: 84.3%

This result should not be presented as successful external validation. The row
audit explains an important cause: several new channel or electrolyte materials
are unseen categories, and each paper omits 10-14 of the model's 25 raw inputs.
The model also severely underpredicts the new long-retention devices. A defensible
next step is to add these rows to a versioned dataset, retrain without placing the
external test rows in the training fold, and evaluate with grouped-by-paper
cross-validation plus a still-untouched later-paper test set.

## Metrics

Because retention spans orders of magnitude, `RMSE_log1p` and factor error are
the primary metrics. Raw-millisecond MAE/RMSE are included but can be dominated
by one long-retention device. Row-level columns include signed error, absolute
error, APE, sMAPE, log error, and symmetric factor error.

## P3HT paper-grouped retraining

`p3ht_validation.py` is a separate, non-destructive experiment. It does not
overwrite the deployed model. The workflow first keeps the 431 P3HT rows from
62 historical paper identifiers. It then narrows training to rows matching the
external material domain when each filter still leaves at least eight papers.
For the current external paper, the retained domain contains 165 rows from 21
papers and matches spin coating, EMIM, and a PVDF-HFP-containing electrolyte.

Candidate selection uses five-fold `GroupKFold` by `Paper_ID`. A complete paper
is therefore confined to either training or validation in each fold. The
external target values are not accepted by the candidate-selection function.
Post-fabrication electrical measurements and gate-pulse measurement protocol
fields are excluded so the model uses material and fabrication information.

The physics-on variant includes the directly calculated ion diffusivity and
viscosity as well as log diffusivity, log viscosity, diffusivity-to-viscosity
ratio, ionic mobility proxy, and a radius-to-diffusion-timescale proxy. The
physics-off ablation removes this entire feature family while keeping identical
paper folds.

Run:

```bash
python -m unittest discover -s tests -v
python p3ht_validation.py \
  --historical-data 'data/통합 data ver.5.xlsx' \
  --external-data external_validation_data.csv \
  --out validation_results
```

The current external P3HT paper contributes three directly reported decay
constants. The physics-on predictions are:

- BF4: actual 275 ms, prediction 1601.1 ms, APE 482.20%
- TFSI: actual 406 ms, prediction 1071.7 ms, APE 163.97%
- MeSO4: actual 1059 ms, prediction 1155.1 ms, APE 9.07%

The three-row MAPE is 218.42%. The 9.07% value is a real row-level result for
the MeSO4 condition and must not be described as the overall external error.
`validation_results/p3ht_external_predictions.csv` preserves all eligible rows
so this distinction can be audited.
