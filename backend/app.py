from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "generation_forecast.csv"
FRONTEND_PATH = BASE / "frontend"

FEATURES = [
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "outage_flag",
]
TARGET = "actual_mwh"

app = FastAPI(
    title="Power Generation Forecasting & Variance Explanation Agent",
    description="A baseline ML model and evidence-based forecast variance analysis.",
    version="1.0.0",
)


class RunRequest(BaseModel):
    date: str | None = None


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = ["date", *FEATURES, "forecast_mwh", "actual_mwh"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    return df


def train_and_evaluate(df: pd.DataFrame) -> dict[str, Any]:
    # Chronological holdout: earlier records train the model, later records test it.
    split_at = max(int(len(df) * 0.8), 1)
    if split_at >= len(df):
        split_at = len(df) - 1
    train_df = df.iloc[:split_at]
    test_df = df.iloc[split_at:]

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(train_df[FEATURES], train_df[TARGET])
    predictions = model.predict(test_df[FEATURES])
    predictions = np.maximum(predictions, 0)

    mae = float(mean_absolute_error(test_df[TARGET], predictions))
    rmse = float(np.sqrt(mean_squared_error(test_df[TARGET], predictions)))
    r2 = float(r2_score(test_df[TARGET], predictions)) if len(test_df) > 1 else None

    # Compare the provided operational forecast to actuals on the same held-out dates.
    operational_mae = float(mean_absolute_error(test_df[TARGET], test_df["forecast_mwh"]))
    return {
        "model": model,
        "split_at": split_at,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae_mwh": round(mae, 3),
        "rmse_mwh": round(rmse, 3),
        "r2": round(r2, 3) if r2 is not None and np.isfinite(r2) else None,
        "provided_forecast_mae_mwh": round(operational_mae, 3),
    }


def explain_variance(row: pd.Series) -> list[str]:
    forecast = float(row["forecast_mwh"])
    actual = float(row["actual_mwh"])
    variance = actual - forecast
    notes: list[str] = []

    if variance > 0:
        notes.append("Actual generation was above the provided forecast.")
    elif variance < 0:
        notes.append("Actual generation was below the provided forecast.")
    else:
        notes.append("Actual generation matched the provided forecast.")

    irradiance = float(row["irradiance_wm2"])
    wind = float(row["wind_speed_ms"])
    temp = float(row["temperature_c"])
    outage = int(row["outage_flag"])

    if irradiance < 500:
        notes.append(f"Irradiance was relatively low ({irradiance:.1f} W/m²), which may reduce solar generation.")
    elif irradiance >= 750:
        notes.append(f"Irradiance was high ({irradiance:.1f} W/m²); this is a potentially favorable solar-generation condition.")

    if temp >= 35:
        notes.append(f"Temperature was high ({temp:.1f} °C); heat-related efficiency loss is a possible factor, depending on the asset.")
    if wind >= 10:
        notes.append(f"Wind speed was relatively high ({wind:.1f} m/s); its impact depends on whether the asset includes wind generation.")
    if outage == 1:
        notes.append("The outage flag is active and may indicate reduced availability or curtailment.")

    if len(notes) == 1:
        notes.append("No strong rule-based driver was identified from the available fields.")
    notes.append("These are heuristic signals, not causal conclusions; confirm with plant logs and asset-specific data.")
    return notes


def row_result(row: pd.Series, ml_prediction: float | None = None) -> dict[str, Any]:
    forecast = float(row["forecast_mwh"])
    actual = float(row["actual_mwh"])
    variance = actual - forecast
    pct = (variance / forecast * 100.0) if forecast else None
    result = {
        "date": row["date"].strftime("%Y-%m-%d"),
        "forecast_mwh": round(forecast, 2),
        "actual_mwh": round(actual, 2),
        "variance_mwh": round(variance, 2),
        "variance_pct": round(pct, 2) if pct is not None else None,
        "irradiance_wm2": round(float(row["irradiance_wm2"]), 1),
        "wind_speed_ms": round(float(row["wind_speed_ms"]), 2),
        "temperature_c": round(float(row["temperature_c"]), 1),
        "outage_flag": int(row["outage_flag"]),
        "explanation": explain_variance(row),
    }
    if ml_prediction is not None:
        result["model_predicted_actual_mwh"] = round(max(float(ml_prediction), 0), 2)
    return result


@app.get("/api/demo")
def demo():
    df = load_data()
    return {
        "case_study": "Power Generation Forecasting & Variance Explanation Agent",
        "status": "running",
        "dataset_rows": int(len(df)),
        "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        "features": FEATURES,
        "target": TARGET,
    }


@app.get("/api/summary")
def summary():
    df = load_data()
    evaluation = train_and_evaluate(df)
    variance = df["actual_mwh"] - df["forecast_mwh"]
    return {
        "records": int(len(df)),
        "total_forecast_mwh": round(float(df["forecast_mwh"].sum()), 2),
        "total_actual_mwh": round(float(df["actual_mwh"].sum()), 2),
        "mean_variance_mwh": round(float(variance.mean()), 2),
        "mean_absolute_variance_mwh": round(float(variance.abs().mean()), 2),
        "below_forecast_days": int((variance < 0).sum()),
        "above_forecast_days": int((variance > 0).sum()),
        "evaluation": {k: v for k, v in evaluation.items() if k != "model"},
    }


@app.get("/api/records")
def records():
    df = load_data()
    evaluation = train_and_evaluate(df)
    model = evaluation["model"]
    predictions = np.maximum(model.predict(df[FEATURES]), 0)
    return {"records": [row_result(row, pred) for (_, row), pred in zip(df.iterrows(), predictions)]}


@app.post("/api/run")
def run_case(request: RunRequest):
    df = load_data()
    evaluation = train_and_evaluate(df)
    model = evaluation["model"]

    if request.date:
        selected = df[df["date"].dt.strftime("%Y-%m-%d") == request.date]
        if selected.empty:
            raise HTTPException(status_code=404, detail="Date not found in the bundled dataset.")
        row = selected.iloc[0]
        prediction = float(model.predict(row[FEATURES].to_frame().T)[0])
        result = row_result(row, prediction)
    else:
        row = df.iloc[-1]
        prediction = float(model.predict(row[FEATURES].to_frame().T)[0])
        result = row_result(row, prediction)

    return {
        "result": result,
        "evaluation": {k: v for k, v in evaluation.items() if k != "model"},
        "method": "Standardized Linear Regression with chronological 80/20 holdout evaluation",
        "limitations": [
            "The dataset contains only 60 dummy daily records.",
            "The model is a baseline, not a production-grade energy forecasting system.",
            "Weather and outage rules indicate possible drivers but do not establish causality.",
            "More asset metadata, historical seasonality, forecast weather, and plant maintenance logs are needed for production use.",
        ],
    }


if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")


@app.get("/")
def home():
    index = FRONTEND_PATH / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(index)
