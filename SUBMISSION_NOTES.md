# Submission Notes

## Project
Power Generation Forecasting & Variance Explanation Agent

## How to run
See `README.md`. Run `uvicorn backend.app:app --reload` from the project root and open `http://127.0.0.1:8000`.

## Evaluation results
Fill this section with the values returned by your own local run at `/api/summary`:
- Model MAE (MWh):
- Model RMSE (MWh):
- Model R²:
- Provided forecast MAE on the same holdout (MWh):

## Demo
- Local URL:
- GitHub repository:
- Railway deployment:

## Design choices
- Standardized Linear Regression chosen as an interpretable baseline for a small dataset.
- Chronological holdout used instead of a random split.
- Provided operational forecast is evaluated separately from the ML model.
- Explanation layer uses transparent rules and reports possible signals rather than causal claims.

## Limitations
The dataset is small and synthetic. Model performance is not evidence of production readiness. Real deployment needs asset-specific context, larger historical data, reliable weather forecasts, maintenance/outage records, and ongoing monitoring.
