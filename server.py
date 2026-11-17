import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
from datetime import datetime, timedelta, timezone
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Setup Open-Meteo client
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
# Use the historical weather archive API instead of the standard endpoint for long history
wx_url = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 22.5726, 88.3639  # Kolkata

def fetch_hourly(url, params, var_names):
    # Indexing [0] to extract the first response from the list returned
    hourly = openmeteo.weather_api(url, params=params)[0].Hourly()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, name in enumerate(var_names):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)

# 2. Fetch 400 Days of Historical Air Quality + Historical Weather
today = datetime.now(timezone.utc).date()
start_400d = (today - timedelta(days=400)).isoformat()
end_today = today.isoformat()

aq_vars = ["pm10", "pm2_5"]
wx_vars = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]

df_aq_hist = fetch_hourly(aq_url, {
    "latitude": LAT, "longitude": LON, "hourly": aq_vars,
    "start_date": start_400d, "end_date": end_today, "timezone": "Asia/Kolkata"
}, aq_vars)

df_wx_hist = fetch_hourly(wx_url, {
    "latitude": LAT, "longitude": LON, "hourly": wx_vars,
    "start_date": start_400d, "end_date": end_today, "timezone": "Asia/Kolkata"
}, wx_vars)

df_hist = pd.merge(df_aq_hist, df_wx_hist, on="date").dropna().reset_index(drop=True)

# 3. Fetch Recent 2 Days + Future 7 Days (Air Quality + Weather Forecasts)
# Use forecast API for short-term weather
forecast_wx_url = "https://api.open-meteo.com/v1/forecast"

df_aq_fc = fetch_hourly(aq_url, {
    "latitude": LAT, "longitude": LON, "hourly": aq_vars,
    "past_days": 2, "forecast_days": 7, "timezone": "Asia/Kolkata"
}, aq_vars)

df_wx_fc = fetch_hourly(forecast_wx_url, {
    "latitude": LAT, "longitude": LON, "hourly": wx_vars,
    "past_days": 2, "forecast_days": 7, "timezone": "Asia/Kolkata"
}, wx_vars)

df_fc = pd.merge(df_aq_fc, df_wx_fc, on="date").dropna().reset_index(drop=True)

now_utc = pd.Timestamp.now(tz="UTC").floor("h")
past_recent = df_fc[df_fc["date"] <= now_utc].copy()
future_7d = df_fc[df_fc["date"] > now_utc].copy().reset_index(drop=True)

# 4. Feature Engineering (Lags + Weather)
df_hist["pm10_lag_1"] = df_hist["pm10"].shift(1)
df_hist["pm10_lag_24"] = df_hist["pm10"].shift(24)
df_hist["pm10_roll_6"] = df_hist["pm10"].shift(1).rolling(6).mean()

df_hist["pm2_5_lag_1"] = df_hist["pm2_5"].shift(1)
df_hist["pm2_5_lag_24"] = df_hist["pm2_5"].shift(24)
df_hist["pm2_5_roll_6"] = df_hist["pm2_5"].shift(1).rolling(6).mean()

df_hist["pm10_momentum"] = df_hist["pm10"] - df_hist["pm10"].shift(1)
df_hist["pm2_5_momentum"] = df_hist["pm2_5"] - df_hist["pm2_5"].shift(1)
df_hist["hour"] = df_hist["date"].dt.tz_convert("Asia/Kolkata").dt.hour

df_hist["pm10_diff_1"] = df_hist["pm10"].shift(-1) - df_hist["pm10"]
df_hist["pm2_5_diff_1"] = df_hist["pm2_5"].shift(-1) - df_hist["pm2_5"]
train_df = df_hist.dropna().reset_index(drop=True)

features = [
    "pm10", "pm2_5",
    "pm10_lag_1", "pm10_lag_24", "pm10_roll_6",
    "pm2_5_lag_1", "pm2_5_lag_24", "pm2_5_roll_6",
    "pm10_momentum", "pm2_5_momentum", "hour",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
]

# 5. Train Weather-Aware Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(train_df[features], train_df[["pm10_diff_1", "pm2_5_diff_1"]])

# 6. Recursive Forecast Loop Feeding Real Weather Forecasts at Every Step
pm10_buf = list(past_recent["pm10"].tail(30))
pm25_buf = list(past_recent["pm2_5"].tail(30))
my_pm10_preds, my_pm25_preds = [], []

for idx, row_wx in future_7d.iterrows():
    local_hour = row_wx["date"].tz_convert("Asia/Kolkata").hour
    feat_row = pd.DataFrame([{
        "pm10": pm10_buf[-1],
        "pm2_5": pm25_buf[-1],
        "pm10_lag_1": pm10_buf[-2],
        "pm10_lag_24": pm10_buf[-25],
        "pm10_roll_6": sum(pm10_buf[-7:-1]) / 6,
        "pm2_5_lag_1": pm25_buf[-2],
        "pm2_5_lag_24": pm25_buf[-25],
        "pm2_5_roll_6": sum(pm25_buf[-7:-1]) / 6,
        "pm10_momentum": pm10_buf[-1] - pm10_buf[-2],
        "pm2_5_momentum": pm25_buf[-1] - pm25_buf[-2],
        "hour": local_hour,
        "temperature_2m": row_wx["temperature_2m"],
        "relative_humidity_2m": row_wx["relative_humidity_2m"],
        "wind_speed_10m": row_wx["wind_speed_10m"],
        "precipitation": row_wx["precipitation"],
    }])

    diff_pm10, diff_pm25 = model.predict(feat_row)[0]
    next_pm10 = max(0.0, pm10_buf[-1] + diff_pm10)
    next_pm25 = max(0.0, pm25_buf[-1] + diff_pm25)

    my_pm10_preds.append(round(next_pm10, 1))
    my_pm25_preds.append(round(next_pm25, 1))

    pm10_buf.append(next_pm10)
    pm25_buf.append(next_pm25)

# 7. Clean Calendar-Date Summary Table
comp = pd.DataFrame({
    "kolkata_time": future_7d["date"].dt.tz_convert("Asia/Kolkata"),
    "om_pm2_5": future_7d["pm2_5"].round(1),
    "my_pm2_5": my_pm25_preds,
    "wind_kmh": future_7d["wind_speed_10m"].round(1),
    "rain_mm": future_7d["precipitation"].round(1),
})
comp["calendar_date"] = comp["kolkata_time"].dt.date

print(f"Weather-Aware Overall Difference (MAE): {mean_absolute_error(comp["om_pm2_5"], comp["my_pm2_5"]):.2f} µg/m³\n")
daily = comp.groupby("calendar_date")[["om_pm2_5", "my_pm2_5", "wind_kmh", "rain_mm"]].mean().round(1)
daily["pm2_5_diff"] = (daily["my_pm2_5"] - daily["om_pm2_5"]).round(1)
print(daily.to_string())

# 8. Matplotlib Visualization (PM2.5 Curves + Wind/Rain Drivers)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

# Top panel: PM2.5 Forecast Comparison
ax1.plot(comp["kolkata_time"], comp["om_pm2_5"], label="Open-Meteo Official Forecast", color="#1f77b4", linewidth=2.2)
ax1.plot(comp["kolkata_time"], comp["my_pm2_5"], label="My Weather-Aware Random Forest", color="#ff7f0e", linestyle="--", linewidth=2.2)
ax1.fill_between(comp["kolkata_time"], comp["om_pm2_5"], comp["my_pm2_5"], color="gray", alpha=0.15, label="Forecast Gap")
ax1.set_title("Kolkata PM2.5 Forecast Comparison: Weather-Aware Random Forest vs. Open-Meteo", fontsize=13, fontweight="bold")
ax1.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=10)

# Bottom panel: Wind Speed & Rain (Explains why PM2.5 drops or spikes)
ax2.plot(comp["kolkata_time"], comp["wind_kmh"], label="Wind Speed (km/h)", color="#2ca02c", linewidth=1.8)
ax2.bar(comp["kolkata_time"], comp["rain_mm"] * 10, width=0.03, label="Precipitation (scaled x10)", color="#17becf", alpha=0.6)
ax2.set_ylabel("Wind (km/h)", fontsize=11)
ax2.set_xlabel("Kolkata Local Time (IST)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", fontsize=10)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
plt.tight_layout()
plt.show()