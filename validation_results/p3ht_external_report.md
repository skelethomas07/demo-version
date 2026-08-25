# P3HT paper-grouped external validation

The model candidate was selected only by paper-grouped cross-validation on historical P3HT data. External targets were used after model selection.

## Historical grouped validation

Physics-on best candidate: et_leaf2_mf050_s3
Physics-off best candidate: et_leaf4_mf065_s10
Physics-on mean MAE in log1p space: 1.8276
Physics-off mean MAE in log1p space: 1.8345

## External P3HT result

All eligible rows: 3 from 1 paper
All-row MAPE: 218.42%
All-row median factor error: 2.640x
Highest input-similarity third MAPE: 163.97%
Highest input-similarity third rows: wan2026_tfsi

The highest-similarity subset is selected from materials and process inputs only. It is a scope-limited case result, not the overall external error.

## Row results

| validation_id | Anion | Tau_ms | pred_tau_ms | ape_percent | pred_tau_ms_physics_off | physics_off_ape_percent | input_similarity_score | physics_imputation_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wan2026_bf4 | BF4 | 275.0000 | 1601.0583 | 482.2030 | 1418.1720 | 415.6989 | 0.7785 | cation+anion |
| wan2026_tfsi | TFSI | 406.0000 | 1071.7355 | 163.9743 | 1073.4692 | 164.4013 | 0.8890 | cation+anion |
| wan2026_meso4 | MeSO4 | 1059.0000 | 1155.0577 | 9.0706 | 1413.5515 | 33.4798 | 0.7721 | anion |
