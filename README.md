# Synapse Retention Engine Beta

A Streamlit beta web app for condition-to-retention prediction in electrolyte-gated synaptic transistor datasets.

## What changed in this build

- `wt` is treated as a numeric weight-ratio field.
- `polymer` is treated as a categorical polymer/matrix field.
- Rows where `wt` and `polymer` were swapped are corrected during preprocessing.
- Case-only duplicates and common notation variants are normalized before training and inference.
  - Examples: `au`, `Au`, `gold` -> `Au`; `cr_au` -> `Cr/Au`; `Drop_casting` -> `Drop-casting`.
- `Spin_RPM` is only enabled in the UI for spin-based processes.
  - For non-spin processes, it is passed as missing and the model uses the pipeline's imputation logic.
- Additional process-aware features are generated:
  - `is_spin_process`
  - `spin_rpm_effective`
  - `log1p_spin_rpm_effective`

## Repository structure

```text
app.py
model_utils.py
train_model.py
requirements.txt
runtime.txt
.streamlit/config.toml
models/retention_model.joblib
models/input_schema.json
data/.gitkeep
```

The deployed app only needs the files above. The raw spreadsheet is not required for prediction.

## Streamlit Cloud

Use:

```text
https://github.com/<username>/<repo>/blob/main/app.py
```

or the interactive picker:

```text
Repository: <username>/<repo>
Branch: main
Main file path: app.py
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retraining

Place your spreadsheet at `data/retention_dataset.xlsx`, then run:

```bash
python train_model.py --data data/retention_dataset.xlsx --out models/retention_model.joblib
```

Then commit the updated `models/retention_model.joblib` and `models/input_schema.json`.
