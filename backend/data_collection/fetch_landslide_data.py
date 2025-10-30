import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
import time
import concurrent.futures

# -----------------------------
# 1️⃣ SETUP LOGGING
# -----------------------------
# Configure logging to write to a file and stream to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_fetch.log"),
        logging.StreamHandler()
    ]
)

logging.info("-------------------------------------------------")
logging.info("Landslide EVENT-BASED Data Fetcher Script STARTED")
logging.info("-------------------------------------------------")


# -----------------------------
# 2️⃣ CONFIGURATION
# -----------------------------
# General config
days_window = 30  # We will look at a 30-day window
api_timeout = 15  # seconds
api_retries = 2
MAX_WORKERS = 20  # Number of parallel threads for data fetching

# --- FIX: Use timezone-aware datetime ---
# We don't need the 2-day offset, as we are looking at historical events
# However, we'll use a start date to filter landslides
MIN_LANDSLIDE_DATE = "2008-01-01"


# -----------------------------
# 3️⃣ API FUNCTIONS (Now with error handling)
# -----------------------------
def get_openmeteo_data(lat, lon, start_date, end_date):
    """Fetches HOURLY weather data from Open-Meteo with retries and error handling."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,relative_humidity_2m"
        f"&timezone=auto"
    )
    
    for attempt in range(api_retries):
        try:
            r = requests.get(url, timeout=api_timeout)
            r.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            data = r.json()
            
            df = pd.DataFrame({
                "timestamp": data["hourly"]["time"],
                "temperature": data["hourly"]["temperature_2m"],
                "precipitation": data["hourly"]["precipitation"],
                "humidity": data["hourly"]["relative_humidity_2m"]
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lat"] = lat
            df["lon"] = lon
            return df
        
        except requests.exceptions.HTTPError as e:
            if r.status_code == 400:
                logging.warning(f"(Open-Meteo) No data for ({lat}, {lon}). Skipping point.")
            else:
                logging.warning(f"[Open-Meteo] HTTP Error for ({lat}, {lon}): {e}. Response: {r.text[:100]}")
            break # Don't retry on HTTP 4xx/5xx, it's a permanent-ish error
        except requests.exceptions.RequestException as e:
            logging.warning(f"[Open-Meteo] Request failed for ({lat}, {lon}): {e}. Retrying ({attempt+1}/{api_retries})...")
            time.sleep(1 * (attempt + 1)) # Exponential backoff
        except (KeyError, requests.exceptions.JSONDecodeError) as e:
            logging.warning(f"(Open-Meteo) Data missing at ({lat}, {lon}).")
            break # Data schema missing = point w/ no data
        except Exception as e:
            logging.error(f"[Open-Meteo] Unexpected error for ({lat}, {lon}): {e}")
            break

    # Return an empty DataFrame if all attempts fail
    return pd.DataFrame(columns=["timestamp", "temperature", "precipitation", "humidity", "lat", "lon"])

def get_openmeteo_daily_env(lat, lon, start_date, end_date):
    """
    Fetches DAILY environmental data from Open-Meteo Archive.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=evapotranspiration_fao_56,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm&timezone=auto"
    )
    
    for attempt in range(api_retries):
        try:
            r = requests.get(url, timeout=api_timeout)
            r.raise_for_status()
            data = r.json()["daily"]
            df = pd.DataFrame({
                "timestamp": data["time"],
                "et_fao_56": data["evapotranspiration_fao_56"],
                "soil_temp_0_7cm": data["soil_temperature_0_to_7cm"],
                "soil_moisture_0_to_7cm": data["soil_moisture_0_to_7cm"]
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["lat"] = lat
            df["lon"] = lon
            return df

        except requests.exceptions.HTTPError as e:
            if r.status_code == 400:
                logging.warning(f"(DailyEnv) No data for ({lat}, {lon}). Skipping point.")
            else:
                logging.warning(f"[Open-Meteo DailyEnv] HTTP Error for ({lat}, {lon}): {e}. Response: {r.text[:100]}")
            break
        except requests.exceptions.RequestException as e:
            logging.warning(f"[Open-Meteo DailyEnv] Request failed for ({lat}, {lon}): {e}. Retrying ({attempt+1}/{api_retries})...")
            time.sleep(1 * (attempt + 1))
        except (KeyError, requests.exceptions.JSONDecodeError) as e:
            logging.warning(f"(DailyEnv) Data missing at ({lat}, {lon}).")
            break
        except Exception as e:
            logging.error(f"[Open-Meteo DailyEnv] Unexpected error for ({lat}, {lon}): {e}")
            break
            
    return pd.DataFrame(columns=["timestamp", "et_fao_56", "soil_temp_0_7cm", "soil_moisture_0_to_7cm", "lat", "lon"])

def get_landslide_data_india():
    """
    Fetches historical landslide data from your local CSV file.
    """
    logging.info("Fetching historical landslide catalog from local file...")
    try:
        # --- NEW LOCAL FILE PATH ---
        url = "backend/data_collection/final_landslide_event_features.csv"
        df = pd.read_csv(url)
        
        logging.info("Successfully loaded local CSV.")
        
        # --- REMOVED India filter ---
        # (Filtering logic removed as requested)

        # Convert 'event_date' (which is text) to datetime
        # The new format is often '2015/07/25 00:00:00'
        df["date"] = pd.to_datetime(df["event_date"], errors="coerce")
        
        # --- Add a unique ID for merging ---
        # Use OBJECTID if it exists, otherwise reset index
        if "OBJECTID" in df.columns:
            df = df.rename(columns={"OBJECTID": "landslide_id"})
        else:
            df = df.reset_index().rename(columns={"index": "landslide_id"})
        
        # Keep relevant columns
        # Ensure 'country' column exists by renaming 'country_name' if present
        if "country_name" in df.columns and "country" not in df.columns:
            df = df.rename(columns={"country_name": "country"})
            
        # Select columns, handling potential missing ones
        required_cols = ["landslide_id", "date", "lat", "lon"]
        optional_cols = ["country", "landslide_category"]
        final_cols = required_cols[:]
        for col in optional_cols:
            if col in df.columns:
                final_cols.append(col)

        df = df[final_cols]
        
        # --- EDIT: Explicitly drop rows with no date (as requested) ---
        # This handles cases where event_date was null or unparseable
        df = df.dropna(subset=["date"])
        
        # Drop rows with no location
        df = df.dropna(subset=["lat", "lon"])
        
        # Filter for dates Open-Meteo has good data for
        df = df[df["date"] >= pd.to_datetime(MIN_LANDSLIDE_DATE)].reset_index(drop=True)
        
        logging.info(f"Successfully filtered {len(df)} historical landslides (since {MIN_LANDSLIDE_DATE}).")
        return df
    except FileNotFoundError:
        logging.error(f"Failed to fetch historical landslide data: File not found at {url}")
        return pd.DataFrame(columns=["landslide_id", "date", "country", "lat", "lon", "landslide_category"])
    except Exception as e:
        logging.error(f"Failed to process historical landslide data: {e}")
        return pd.DataFrame(columns=["landslide_id", "date", "country", "lat", "lon", "landslide_category"])

# -----------------------------
# 4️⃣ NEW: MERGE AND AGGREGATE HELPER
# -----------------------------
def merge_and_aggregate(weather_df, env_df):
    """Merges and aggregates data for a single period."""
    # Initialize output with NaNs for all metrics
    agg_data = {k: np.nan for k in METRIC_KEYS}

    if weather_df.empty and env_df.empty:
        return agg_data

    # Ensure timestamp columns are datetime before using .dt
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
    env_df['timestamp'] = pd.to_datetime(env_df['timestamp'], errors='coerce')

    # --- Convert to daily data ---
    weather_df['date'] = weather_df['timestamp'].dt.date
    env_df['date'] = env_df['timestamp'].dt.date
    
    # Drop rows where date conversion failed
    weather_df = weather_df.dropna(subset=['date'])
    env_df = env_df.dropna(subset=['date'])
    
    daily_weather = pd.DataFrame(columns=['date', 'lat', 'lon'])
    if not weather_df.empty:
        daily_weather = weather_df.groupby(['date', 'lat', 'lon']).agg(
            temperature=('temperature', 'mean'),
            precipitation=('precipitation', 'sum'),
            humidity=('humidity', 'mean')
        ).reset_index()

    # Merge
    if not daily_weather.empty and not env_df.empty:
        df = daily_weather.merge(env_df, on=["date", "lat", "lon"], how="outer")
    elif not daily_weather.empty:
        df = daily_weather
    elif not env_df.empty:
        df = env_df
    else:
        return agg_data

    # Fill computed metrics where columns exist
    if "temperature" in df:
        agg_data["temperature_mean"] = df["temperature"].mean()
    if "precipitation" in df:
        agg_data["precipitation_sum"] = df["precipitation"].sum()
    if "humidity" in df:
        agg_data["humidity_mean"] = df["humidity"].mean()
    if "soil_moisture_0_to_7cm" in df:
        agg_data["soil_moisture_mean"] = df["soil_moisture_0_to_7cm"].mean()
    if "et_fao_56" in df:
        agg_data["et_fao_56_mean"] = df["et_fao_56"].mean()
    if "soil_temperature_0_to_7cm" in df:
        agg_data["soil_temp_mean"] = df["soil_temperature_0_to_7cm"].mean()
    
    return agg_data

# Metric keys to always include in outputs
METRIC_KEYS = [
    "temperature_mean",
    "precipitation_sum",
    "humidity_mean",
    "soil_moisture_mean",
    "et_fao_56_mean",
    "soil_temp_mean",
]

# -----------------------------
# 5️⃣ NEW: HELPER FUNCTION FOR CONCURRENCY
# -----------------------------
def fetch_data_for_event(task_data):
    """
    Worker function to fetch all data for a single landslide event.
    """
    landslide_id, lat, lon, event_date = task_data
    
    # --- Period 1: "Event Week" (30 days leading up to and including the event) ---
    event_start = event_date - timedelta(days=days_window - 1)
    event_end = event_date
    
    event_weather_df = get_openmeteo_data(lat, lon, event_start, event_end)
    event_env_df = get_openmeteo_daily_env(lat, lon, event_start, event_end)
    
    # --- Period 2: "Prior Week" (the 30 days *before* the event week) ---
    prior_start = event_start - timedelta(days=days_window)
    prior_end = event_start - timedelta(days=1)
    
    prior_weather_df = get_openmeteo_data(lat, lon, prior_start, prior_end)
    prior_env_df = get_openmeteo_daily_env(lat, lon, prior_start, prior_end)

    # --- Aggregate data for both periods ---
    event_week_data = merge_and_aggregate(event_weather_df, event_env_df)
    prior_week_data = merge_and_aggregate(prior_weather_df, prior_env_df)

    # --- Create the final feature row ---
    final_row = {
        "landslide_id": landslide_id,
        "lat": lat,
        "lon": lon,
        "event_date": event_date.strftime("%Y-%m-%d")
    }
    
    # Add features and deltas
    for key in METRIC_KEYS:
        event_val = event_week_data.get(key)
        prior_val = prior_week_data.get(key)
        
        final_row[f"{key}_event_week"] = event_val
        final_row[f"{key}_prior_week"] = prior_val
        
        # Calculate delta
        delta = np.nan
        if pd.notna(event_val) and pd.notna(prior_val):
            delta = event_val - prior_val
        elif pd.notna(event_val):
            delta = event_val # No prior data, delta is just the event value
            
        final_row[f"{key}_delta"] = delta
        
        # Additional derived features
        # change per day normalized by window size
        final_row[f"{key}_change_per_day"] = (delta / days_window) if pd.notna(delta) else np.nan
        
        # percent change relative to prior period
        pct_change = np.nan
        if pd.notna(event_val) and pd.notna(prior_val) and prior_val not in [0, 0.0]:
            pct_change = (event_val - prior_val) / prior_val
        final_row[f"{key}_percent_change"] = pct_change
        
        # ratio event to prior
        ratio = np.nan
        if pd.notna(event_val) and pd.notna(prior_val) and prior_val not in [0, 0.0]:
            ratio = event_val / prior_val
        final_row[f"{key}_ratio_event_to_prior"] = ratio

    return final_row


# -----------------------------
# 6️⃣ MAIN EXECUTION LOOP
# -----------------------------
def main():
    # 1. Get all historical landslides
    landslides_df = get_landslide_data_india()
    if landslides_df.empty:
        logging.error("No landslides found. Exiting.")
        return
        
    # 2. Prepare list of tasks for the thread pool
    tasks = []
    for row in landslides_df.itertuples():
        # Ensure lat/lon are valid floats
        try:
            lat = float(row.lat)
            lon = float(row.lon)
            tasks.append((row.landslide_id, lat, lon, row.date))
        except (ValueError, TypeError):
            logging.warning(f"Skipping landslide ID {row.landslide_id} due to invalid lat/lon.")
            
    total_tasks = len(tasks)
    logging.info(f"Starting data collection for {total_tasks} landslide events using {MAX_WORKERS} workers...")
    
    results = []
    
    # 3. Run tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(fetch_data_for_event, task): task for task in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
            task_data = future_to_task[future]
            landslide_id = task_data[0]
            try:
                result_row = future.result()
                results.append(result_row)
                logging.info(f"Processed event {i+1}/{total_tasks} (ID: {landslide_id})... COMPLETE.")
            except Exception as e:
                logging.error(f"Event (ID: {landslide_id}) failed with an unexpected error: {e}", exc_info=True)

    logging.info("...Bulk data collection finished.")
    
    # 4. Create and save final DataFrame
    if not results:
        logging.error("No results were generated. Exiting.")
        return
        
    final_df = pd.DataFrame(results)
    
    # 5. Merge original landslide info (like category) back in
    # Check which optional columns we actually have
    optional_cols_to_merge = []
    if "landslide_category" in landslides_df.columns:
        optional_cols_to_merge.append("landslide_category")
    if "country" in landslides_df.columns:
        optional_cols_to_merge.append("country")

    if optional_cols_to_merge:
        final_df = final_df.merge(
            landslides_df[["landslide_id"] + optional_cols_to_merge],
            on="landslide_id",
            how="left"
        )
    
    output_filename = "final_landslide_event_features.csv"
    try:
        final_df.to_csv(output_filename, index=False)
        logging.info(f"...Successfully saved all event features to {output_filename}.")
    except Exception as e:
        logging.error(f"Failed to save CSV file: {e}")

    logging.info("\n✅ Final Event-Based Features Sample:\n" + final_df.head().to_string())
    logging.info("-------------------------------------------------")
    logging.info("✅ Script Finished")
    logging.info("-------------------------------------------------")


if __name__ == "__main__":
    main()

