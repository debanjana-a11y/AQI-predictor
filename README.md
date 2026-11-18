
# Hyper-Local AQI & PM2.5 Forecasting Engine (Kolkata)

A machine learning air-quality forecasting and hyper-local calibration pipeline built with Random Forest Regression, OpenAQ ground-station telemetry, and Open-Meteo atmospheric data.

Trained on 400 days of real physical sensor logs from the West Bengal Pollution Control Board (WBPCB) station at Rabindra Sarobar, Kolkata (ID: 10904), this model predicts short-term PM2.5 levels with ±3.61 µg/m³ Mean Absolute Error (MAE). That reduces hyper-local street-level error by 90.1% compared with the raw regional model's 36.51 µg/m³ MAE.

<img width="2100" height="1200" alt="last_30_days_comparison" src="https://github.com/user-attachments/assets/b2618187-9aae-4320-baf1-3e631f185528" />


## Why This Project Exists

Global air-quality APIs such as Open-Meteo and CAMS rely on coarse 15–40 km atmospheric grid models. Although they perform well for regional weather systems such as Bay of Bengal cyclones, coarse grids miss neighborhood traffic corridors, construction dust, and local winter inversions.

This project addresses that gap by:

1. Ingesting real physical CPCB/WBPCB ground-station logs directly from OpenAQ's public S3 archive, with no API key required.
2. Engineering time-series momentum and cyclical features, including `lag_1`, `lag_24`, `roll_6`, `momentum`, and local IST hour, to capture daily urban emission rhythms.
3. Predicting hourly deltas (`target - current`) rather than raw concentrations, reducing tree-model ceiling effects during severe winter pollution spikes.

## Limitation
That revealed that while my local Random Forest excels at 1–6 hour hyper-local nowcasting, global atmospheric models (Open Meteo) remain stronger for multi-day weather fronts like Bay of Bengal cyclones.

## Benchmark Results

The model was evaluated chronologically across 720 held-out test hours against physical WBPCB sensor readings. A total of 8,828 hourly records were ingested.

| Model / source | Forecast horizon | Mean Absolute Error (MAE) | Accuracy improvement |
|---|---:|---:|---:|
| Open-Meteo (raw regional grid) | 1 hour ahead | 36.51 µg/m³ | Baseline |
| Hyper-local Random Forest (ours) | 1 hour ahead | 3.61 µg/m³ | 90.1% error reduction |

The generated comparison plot is saved automatically as `last_30_days_comparison.png`.

## Project Architecture

```text
├── server.py                      # End-to-end ingestion, training, backtesting, and plotting
├── station_10904_400d.csv         # Local cache of 400 days of WBPCB hourly sensor data
├── last_30_days_comparison.png    # Ground truth vs. Open-Meteo vs. Random Forest report
└── README.md                      # Project documentation
```

## Pipeline Workflow

1. **Ground-truth ingestion:** Uses `boto3` to anonymously fetch 400 days of daily `.csv.gz` logs from `s3://openaq-data-archive` for Rabindra Sarobar (`locationid=10904`) and caches them locally.
2. **Continuous grid alignment:** Resamples irregular sensor logs onto a continuous UTC/IST hourly index, interpolating short 1–3 hour hardware dropouts while preserving raw values for evaluation.
3. **Feature engineering:** Computes `lag_1`, `lag_24`, `roll_6` (6-hour trailing average), `momentum` (1-hour rate of change), and `hour` (Asia/Kolkata).
4. **Chronological train/test split:** Trains a 200-tree `RandomForestRegressor` with `max_depth=12` on the first approximately 370 days and evaluates strictly on the final 30 days to avoid data leakage.
5. **Visual verification:** Compares model predictions with physical sensor readings and Open-Meteo's historical baseline using Matplotlib.

## Quickstart

### Prerequisites

- Python 3.10 or newer
- A virtual environment such as `venv` or Conda

### Installation

```bash
git clone <your-repo-url>
cd AQI-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn matplotlib boto3 openmeteo-requests requests-cache retry-requests
```

### Run the Pipeline

```bash
python3 server.py
```

- On the first run, the script downloads approximately 392 daily sensor files from the OpenAQ S3 archive and caches them to `station_10904_400d.csv`.
- On subsequent runs, it loads directly from the cached CSV and completes training, evaluation, and chart generation in seconds.

## Future Roadmap

- **Multi-horizon blending:** Combine this hyper-local Random Forest model (1–6 hour nowcasting) with Open-Meteo's weather-driven atmospheric model (2–5 day cyclone and rain forecasting).
- **Indian CPCB NAQI conversion:** Map predicted PM2.5 and PM10 concentrations to official 24-hour Indian National Air Quality Index categories, ranging from 0–500.
- **REST API and automated alerts:** Wrap the trained model in FastAPI and add scheduled WhatsApp or webhook alerts for construction-dust compliance (WBPCB/GRAP) and smart HVAC ventilation control.
