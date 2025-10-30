import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm


def get_meteo_data(lat: float, lon: float, date: datetime):
    """
    Fetch daily meteo data (rainfall, temperature, soil moisture) for a given date & location
    using Open-Meteo API. Mirrors variables used in backend/data_collection/test.py
    """
    date_str = date.strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily=precipitation_sum,temperature_2m_mean,soil_moisture_0_to_10cm_mean"
        f"&timezone=auto"
    )
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        return None
    data = response.json()
    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        return None
    daily = data["daily"]
    return {
        "rainfall": daily["precipitation_sum"][0] if daily.get("precipitation_sum") else None,
        "temperature": daily["temperature_2m_mean"][0] if daily.get("temperature_2m_mean") else None,
        "soil_moisture": daily.get("soil_moisture_0_to_10cm_mean", [None])[0],
    }


def main():
    # Load event features to source lat/lon and event_date per landslide_id
    src_path = "backend/data_collection/final_landslide_event_features.csv"
    df = pd.read_csv(src_path)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            lat = float(row["lat"]) if "lat" in row else float(row["latitude"])  # fallback
            lon = float(row["lon"]) if "lon" in row else float(row["longitude"])  # fallback
        except Exception:
            continue

        event_date_str = str(row["event_date"])[:10]
        try:
            event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
        except Exception:
            continue

        # Negative sample date: the day before the event
        neg_date = event_date - timedelta(days=1)
        neg_ref_date = neg_date - timedelta(days=30)

        current = get_meteo_data(lat, lon, neg_date)
        prev = get_meteo_data(lat, lon, neg_ref_date)
        if not current or not prev:
            continue

        rainfall_delta = (current["rainfall"] or 0) - (prev["rainfall"] or 0)
        temp_delta = (current["temperature"] or 0) - (prev["temperature"] or 0)
        soil_moisture_delta = (current["soil_moisture"] or 0) - (prev["soil_moisture"] or 0)

        avg_change = (rainfall_delta + temp_delta + soil_moisture_delta) / 3
        change_ratio = avg_change / 3
        change_per_day = avg_change / 30

        # Output schema to match backend/data_collection/test.py, with label=0
        records.append({
            "landslide_id": row.get("landslide_id", None),
            "lat": lat,
            "lon": lon,
            "event_date": neg_date.strftime("%Y-%m-%d"),
            "rainfall_delta": rainfall_delta,
            "temp_delta": temp_delta,
            "soil_moisture_delta": soil_moisture_delta,
            "change_ratio": change_ratio,
            "change_per_day": change_per_day,
            "label": 0,
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv("backend/data_collection/false.csv", index=False)
    print(f"✅ false.csv created with {len(out_df)} rows at backend/data_collection/false.csv")


if __name__ == "__main__":
    main()


