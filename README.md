# Synapse Retention Engine Beta v3

A Streamlit web app for condition-to-retention prediction of electrolyte-gated synaptic transistor retention time.

## What this version does

- Material, process, electrolyte, ion, and operation conditions are entered by the user.
- The trained Stacking Ensemble model predicts `Tau_ms`.
- The **Input drivers** tab shows the top 5 input fields that most changed the current prediction.
- The **Recommend conditions** tab keeps engineer-selected inputs fixed and searches adjustable process and operation conditions for a requested `Tau_ms`.
- The five closest candidates show predicted retention, target difference, changed conditions, and local input-driver rankings.

## Target-driven condition recommendation

1. Enter the material, ion, process, and operation values that describe the planned device.
2. Enter the target retention time in milliseconds.
3. Select only the controllable fields the engine may change.
4. Review or edit the minimum and maximum search range for each selected condition.
5. Run the search to evaluate 3,000 condition combinations inside both the engineer-approved ranges and the model's recorded training ranges.
6. Review the top five candidates and their input drivers before deciding which experiment to run.

The recommender is a model-based design aid, not a physical-validity guarantee. Material and ion descriptors are held fixed, and an engineer must review each proposed combination before fabrication.

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
