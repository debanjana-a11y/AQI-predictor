import os
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load your 400 days of REAL Rabindra Sarobar sensor data
CACHE_CSV = "station_10904_400d.csv"
LAT, LON = 22.5111, 88.3514  # Rabindra Sarobar, Kolkata

df_real = pd.read_csv(CACHE_CSV)
df_real["date"] = pd.to_datetime(df_real["date"], utc=True).dt.floor("h")
df_real = df_real.sort_values("date").drop_duplicates("date").reset_index(drop=True)
print(f"Loaded {len(df_real)} real hourly readings from {CACHE_CSV}.")

# 2. Build Features ONLY from Real Rabindra Sarobar Sensor Data
start_grid = df_real["date"].min()
end_grid = df_real["date"].max()
full_grid = pd.DataFrame({"date": pd.date_range(start_grid, end_grid, freq="h", tz="UTC")})

df = full_grid.merge(df_real, on="date", how="left")
df["real_clean"] = df["real_pm25"].interpolate(limit=3)

df["lag_1"] = df["real_clean"].shift(1)
df["lag_24"] = df["real_clean"].shift(24)
df["roll_6"] = df["real_clean"].shift(1).rolling(6).mean()
df["momentum"] = df["real_clean"] - df["real_clean"].shift(1)
df["hour"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.hour

df["target_1h"] = df["real_pm25"].shift(-1)
df["diff_1h"] = df["real_clean"].shift(-1) - df["real_clean"]

features = ["real_clean", "lag_1", "lag_24", "roll_6", "momentum", "hour"]

# 3. Split: First ~370 Days = Train | Last 30 Days (720 Hours) = Test
test_hours = 24 * 30
train_df = df.iloc[:-test_hours].dropna(subset=features + ["diff_1h"]).reset_index(drop=True)
test_df = df.iloc[-test_hours:].copy().reset_index(drop=True)

model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(train_df[features], train_df["diff_1h"])

test_valid = test_df.dropna(subset=features + ["target_1h"]).copy()
test_valid["my_pred"] = (test_valid["real_clean"] + model.predict(test_valid[features])).clip(lower=0)
test_valid["target_date"] = test_valid["date"] + pd.Timedelta(hours=1)

# 4. Fetch Last 30 Days from Open-Meteo Using Your Exact Jupyter Client
cache_session = requests_cache.CachedSession(".om_clean_cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

om_start = (test_valid["target_date"].min() - timedelta(days=1)).date().isoformat()
om_end = (test_valid["target_date"].max() + timedelta(days=1)).date().isoformat()

om_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
om_params = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": ["pm2_5"],
    "start_date": om_start,
    "end_date": om_end,
    "timezone": "UTC",
}

responses = openmeteo.weather_api(om_url, params=om_params)
response = responses[0]
hourly = response.Hourly()

df_om = pd.DataFrame({
    "target_date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ),
    "om_pm25": hourly.Variables(0).ValuesAsNumpy(),
}).dropna()

# 5. Compare Real Sensor vs. Open-Meteo vs. My Model on Last 30 Days
comp = test_valid.merge(df_om, on="target_date", how="inner").dropna(subset=["target_1h", "my_pred", "om_pm25"])
comp["kolkata_time"] = comp["target_date"].dt.tz_convert("Asia/Kolkata")

om_mae = mean_absolute_error(comp["target_1h"], comp["om_pm25"])
my_mae = mean_absolute_error(comp["target_1h"], comp["my_pred"])

print("\n=== LAST 30 DAYS SCOREBOARD (VS. REAL RABINDRA SAROBAR SENSOR) ===")
print(f"1. Open-Meteo Mean Deviation (MAE): {om_mae:.2f} µg/m³")
print(f"2. My Model Mean Deviation (MAE):   {my_mae:.2f} µg/m³")

# 6. Plot All Three Curves + Mean Deviation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

ax1.plot(comp["kolkata_time"], comp["target_1h"], label="Real Rabindra Sarobar Sensor (Ground Truth)", color="black", linewidth=2.0)
ax1.plot(comp["kolkata_time"], comp["om_pm25"], label=f"Open-Meteo Reported (Mean Deviation: {om_mae:.2f} µg/m³)", color="#1f77b4", alpha=0.85, linewidth=1.6)
ax1.plot(comp["kolkata_time"], comp["my_pred"], label=f"My Model Predicted (Mean Deviation: {my_mae:.2f} µg/m³)", color="#ff7f0e", linestyle="--", linewidth=1.8)

ax1.set_title("Last 30 Days PM2.5: Real Rabindra Sarobar Sensor vs. Open-Meteo vs. My Model", fontsize=13, fontweight="bold")
ax1.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", fontsize=10)

om_err = (comp["om_pm25"] - comp["target_1h"]).abs()
my_err = (comp["my_pred"] - comp["target_1h"]).abs()

ax2.plot(comp["kolkata_time"], om_err, label=f"Open-Meteo Deviation (Avg: {om_mae:.2f})", color="#1f77b4", alpha=0.75)
ax2.plot(comp["kolkata_time"], my_err, label=f"My Model Deviation (Avg: {my_mae:.2f})", color="#ff7f0e", alpha=0.85)
ax2.set_ylabel("Absolute Error (µg/m³)", fontsize=11)
ax2.set_xlabel("Kolkata Local Time (Last 30 Days)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", fontsize=10)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout()
plt.savefig("last_30_days_comparison.png", dpi=150)
print("\nSaved graph to last_30_days_comparison.png!")
plt.show()