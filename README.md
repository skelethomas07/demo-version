# Synapse Retention Engine Beta v3

A Streamlit web app for condition-to-retention prediction of electrolyte-gated synaptic transistor retention time.

## What this version does

- Material, process, electrolyte, ion, and operation conditions are entered by the user.
- The trained Stacking Ensemble model predicts `Tau_ms`.
- The **Input drivers** tab shows the top 5 input fields that most changed the current prediction.

## Input driver method

The driver tab uses local feature-hiding sensitivity:

1. Run the prediction for the current input condition.
2. Hide one raw input field at a time.
3. Rerun the same trained model.
4. Measure the absolute change in predicted `log1p(Tau_ms)`.
5. Normalize the top 5 effects to 100%.

This is a local sensitivity explanation for the current prediction. It is not a causal attribution and not a global feature-importance score.

## Deploy on Streamlit Community Cloud

Use:

```text
Repository: <your GitHub repository>
Branch: main
Main file path: app.py
Python: 3.11
```

Make sure these files exist:

```text
app.py
model_utils.py
requirements.txt
runtime.txt
.streamlit/config.toml
models/retention_model.joblib
models/input_schema.json
```
