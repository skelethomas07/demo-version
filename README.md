# Synapse Retention Engine

A beta web engine for estimating neuromorphic EGST retention time from material, process, electrolyte, and electrical operation conditions.

## What this app does

- Accepts only condition inputs used by the trained model.
- Applies the same preprocessing and physics-informed feature engineering used in the modeling notebook.
- Runs a trained Stacking Ensemble model.
- Returns a new predicted `Tau_ms` for the entered condition vector.

This is not a nearest-neighbor lookup and does not include target search, similar experiment search, or presentation tabs.

## File structure

```text
app.py
model_utils.py
train_model.py
requirements.txt
runtime.txt
.streamlit/config.toml
models/
  retention_model.joblib
  input_schema.json
data/
  .gitkeep
```

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files and folders in this package.
3. In Streamlit Community Cloud, create a new app.
4. Select the repository.
5. Set the main file path to `app.py`.
6. Deploy.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain model

The deployed app does not need the spreadsheet. To retrain, place your spreadsheet here:

```text
data/retention_dataset.xlsx
```

Then run:

```bash
python train_model.py --data data/retention_dataset.xlsx --out models/retention_model.joblib
```

## Important limitation

Predictions are model-based estimates from the literature-trained distribution. They are intended for screening and should be experimentally validated before device-level decisions.
