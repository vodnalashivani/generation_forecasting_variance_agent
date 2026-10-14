# Power Generation Forecasting & Variance Explanation Agent

## Overview
This project analyzes a bundled 60-row dummy dataset, evaluates an interpretable Linear Regression baseline, compares operational forecast values with actual generation, and presents rule-based variance explanation signals in a browser dashboard.

## Features
- FastAPI backend with `/api/demo`, `/api/summary`, `/api/records`, and `/api/run`.
- Chronological 80/20 train/test holdout to reduce temporal leakage.
- Standardized Linear Regression to estimate actual generation from irradiation, wind speed, temperature, and outage status.
- MAE, RMSE, R², and provided-forecast MAE evaluation.
- Absolute and percentage variance analysis.
- Rule-based explanation signals and a responsive HTML/CSS/JavaScript UI.
- Root route serves the frontend for local and Railway deployment.

## Requirements
- Python 3.10+ (Python 3.12 is suitable)
- VS Code
- Internet connection for installing packages and deployment

## Run on Windows
Open PowerShell in the project root folder.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000` in a browser. API documentation is at `http://127.0.0.1:8000/docs`.

If PowerShell blocks activation, run this in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## API examples
- `GET /api/demo`: app and dataset metadata.
- `GET /api/summary`: dataset summary and model evaluation.
- `GET /api/records`: daily rows with variance analysis.
- `POST /api/run`: analyzes the latest row by default or a date supplied as JSON, e.g. `{"date":"2026-08-01"}`.

## Method
Target: `actual_mwh`. Features: `irradiance_wm2`, `wind_speed_ms`, `temperature_c`, and `outage_flag`. The model uses the earliest 80% of the sorted observations for training and the final 20% for evaluation. `forecast_mwh` is kept separate as the existing operational forecast to compare against actual values; it is not used as a model feature.

Variance is defined as `actual_mwh - forecast_mwh`. Percentage variance is variance divided by forecast, multiplied by 100. Rule-based text flags low/high irradiation, high temperature, high wind, and active outage. These are potential signals, not causal explanations.

## Evaluation
The API calculates MAE and RMSE on the held-out dates, along with R² when possible. It also reports the provided forecast's MAE on the same holdout for a like-for-like comparison. Record the values shown by your own local run in the submission notes; do not claim that a baseline is production-ready just because it runs.

## Assumptions and limitations
- The provided data is dummy data with only 60 daily records.
- This is a demonstration baseline, not a validated plant-level forecasting model.
- The fields may not describe a single consistent asset type; wind speed is only meaningful as a direct driver if the asset includes wind generation.
- Weather thresholds are heuristic and may not generalize to other sites or seasons.
- More historical observations, asset capacity/type, forecast weather, curtailment, maintenance, sensor quality, and plant logs are needed before production use.
- No external LLM is required. The explanation layer is deterministic and transparent, so the app runs without API keys. An LLM could be added later behind environment-variable configuration with safeguards against unsupported causal claims.

## Railway deployment
1. Push this folder to a GitHub repository.
2. In Railway, create a project and deploy from the GitHub repository.
3. Railway should detect `requirements.txt` and use the `Procfile` command.
4. Confirm the start command is `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
5. No external API keys are required for this baseline.
6. After deployment, open the generated public domain and test the dashboard and `/api/demo`.
7. If Railway reports a build/start error, inspect the deployment logs and fix the error before submission.

## Suggested submission checklist
- [ ] Source code and bundled dataset included
- [ ] Local dashboard and API tested
- [ ] `/docs` endpoint opens
- [ ] Evaluation values copied from actual local run
- [ ] Screenshots added to `screenshots/` (optional folder)
- [ ] Limitations and assumptions documented
- [ ] GitHub repository URL ready
- [ ] Railway public URL tested and included in final submission
