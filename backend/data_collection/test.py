import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

# -------------------------------
# Load landslide dataset
# -------------------------------
df = pd.read_csv("backend/data_collection/final_landslide_event_features.csv")

# -------------------------------
# Function: Fetch environmental data using Open-Meteo API
# -------------------------------
def get_meteo_data(lat, lon, date):
    """
    Fetch daily meteo data (rainfall, temp, soil moisture) for a given date & location
    using Open-Meteo API.
    """
    date_str = date.strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily=precipitation_sum,temperature_2m_mean,soil_moisture_0_to_10cm_mean"
        f"&timezone=auto"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()

    # Check data presence
    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        return None

    # Extract data
    daily = data["daily"]
    return {
        "rainfall": daily["precipitation_sum"][0] if daily["precipitation_sum"] else None,
        "temperature": daily["temperature_2m_mean"][0] if daily["temperature_2m_mean"] else None,
        "soil_moisture": daily.get("soil_moisture_0_to_10cm_mean", [None])[0]
    }

# -------------------------------
# Iterate through dataset
# -------------------------------
records = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    lat, lon, event_date = row["lat"], row["lon"], str(row["event_date"])[:10]
    event_date = datetime.strptime(event_date, "%Y-%m-%d")
    pre_date = event_date - timedelta(days=30)

    # Fetch environmental data
    current = get_meteo_data(lat, lon, event_date)
    prev = get_meteo_data(lat, lon, pre_date)

    if not current or not prev:
        continue

    # Calculate deltas
    rainfall_delta = current["rainfall"] - prev["rainfall"]
    temp_delta = current["temperature"] - prev["temperature"]
    soil_moisture_delta = (
        (current["soil_moisture"] or 0) - (prev["soil_moisture"] or 0)
    )

    # Compute change metrics
    avg_change = (rainfall_delta + temp_delta + soil_moisture_delta) / 3
    change_ratio = avg_change / 3
    change_per_day = avg_change / 30

    records.append({
        "landslide_id": row["landslide_id"],
        "lat": lat,
        "lon": lon,
        "event_date": event_date,
        "rainfall_delta": rainfall_delta,
        "temp_delta": temp_delta,
        "soil_moisture_delta": soil_moisture_delta,
        "change_ratio": change_ratio,
        "change_per_day": change_per_day
    })

# -------------------------------
# Save to delta.csv
# -------------------------------
delta_df = pd.DataFrame(records)
delta_df.to_csv("delta.csv", index=False)
print(f"✅ Delta dataset created with {len(delta_df)} records → delta.csv")
