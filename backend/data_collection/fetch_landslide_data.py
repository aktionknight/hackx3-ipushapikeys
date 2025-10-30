import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
import time
import concurrent.futures
# import threading  <-- No longer needed

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
# Windows console-safe logging (remove emoji)
logging.info("Landslide Data Fetcher Script STARTED")
logging.info("-------------------------------------------------")


# -----------------------------
# 2️⃣ CONFIGURATION
# -----------------------------
# General config
days_window = 7
api_timeout = 15  # seconds
api_retries = 2 # Back to 2
MAX_WORKERS = 20  # Number of parallel threads for data fetching
grid_step = 2.0 # Step size in degrees for the grid

# --- Remove Semaphore ---
# NASA_API_SEMAPHORE = threading.Semaphore(5) # <-- No longer needed

# --- FIX: Added a 2-day offset ---
# The archive-api can fail if data is "too recent" (e.g., today or yesterday).
# By setting "today" to 2 days ago, we ensure all data is stably archived.
# --- FIX: Use timezone-aware datetime ---
data_end_date = datetime.now(timezone.utc).date() - timedelta(days=2)
logging.info(f"Current UTC date is {datetime.now(timezone.utc).date()}. Setting data end date to {data_end_date} to ensure archive availability.")
today = data_end_date
one_week_ago = today - timedelta(days=days_window)


# ---- NEW: TARGETED COORDINATE GENERATION ----
def generate_target_coords(step):
    """
    Generates a list of (lat, lon) coordinates for specific high-risk zones
    (Kerala and North India) instead of scanning the whole country.
    """
    coords_list = []
    
    # Zone 1: Kerala
    # Bounding box: ~8°N to 13°N, 75°E to 78°E
    kerala_lats = np.arange(8.0, 13.0 + step, step)
    kerala_lons = np.arange(75.0, 78.0 + step, step)
    for lat in kerala_lats:
        for lon in kerala_lons:
            coords_list.append((round(lat, 2), round(lon, 2)))
    
    logging.info(f"Generated {len(coords_list)} points for Kerala zone.")
    
    # Zone 2: North India (Himalayan belt + NE)
    # Bounding box: ~23°N to 37°N, 72°E to 97°E
    north_lats = np.arange(23.0, 37.0 + step, step)
    north_lons = np.arange(72.0, 97.0 + step, step)
    
    north_points_start = len(coords_list)
    for lat in north_lats:
        for lon in north_lons:
            # Simple filter to remove the western desert/plains (less prone)
            if lon > 76 or lat > 30: # Keep Himalayas + NE states
                 coords_list.append((round(lat, 2), round(lon, 2)))
                 
    logging.info(f"Generated {len(coords_list) - north_points_start} points for North India zone.")

    # Remove duplicates just in case of overlap (though unlikely here)
    coords_set = set(coords_list)
    return list(coords_set)

coords = generate_target_coords(grid_step)

logging.info(f"Configuration Loaded: {len(coords)} total filtered grid points for target zones.")
logging.info(f"Will use {MAX_WORKERS} concurrent workers.")
logging.info(f"Today's Date (UTC): {today}")
logging.info(f"One Week Ago Date (UTC): {one_week_ago}")


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

# --- DELETE THE NASA SOIL FUNCTION ---
# def get_nasa_power_soil(lat, lon, start_date, end_date):
#    ... (This function is now removed) ...


def get_openmeteo_daily_env(lat, lon, start_date, end_date):
    """
    Fetches DAILY environmental data from Open-Meteo Archive.
    --- MODIFIED ---
    Now includes Soil Moisture, replacing the NASA API.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        # --- ADDED soil_moisture_0_to_7cm ---
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
                # --- ADDED soil_moisture_0_to_7cm ---
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
            
    # --- Update empty columns ---
    return pd.DataFrame(columns=["timestamp", "et_fao_56", "soil_temp_0_7cm", "soil_moisture_0_to_7cm", "lat", "lon"])

def get_landslide_data_india():
    """Fetches historical landslide data from NASA GLC."""
    logging.info("Fetching historical landslide catalog...")
    try:
        url = "https://data.nasa.gov/resource/h9d8-neg4.csv"
        df = pd.read_csv(url)
        # Filter India only
        df = df[df["country"].str.contains("India", case=False, na=False)]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[["date", "country", "latitude", "longitude", "landslide_category"]]
        logging.info(f"Successfully fetched {len(df)} historical landslides in India.")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch historical landslide data: {e}")
        return pd.DataFrame(columns=["date", "country", "latitude", "longitude", "landslide_category"])

# -----------------------------
# 4️⃣ NEW: HELPER FUNCTION FOR CONCURRENCY
# -----------------------------
def fetch_data_for_point(point_data):
    """
    Worker function to fetch all 4 data pieces for a single coordinate.
    --- SIMPLIFIED ---
    """
    lat, lon, today, one_week_ago = point_data
    
    # --- Current Data (Yesterday to Today) ---
    start_curr = today - timedelta(days=1)
    end_curr = today
    current_weather_df = get_openmeteo_data(lat, lon, start_curr, end_curr)
    # current_soil_df = get_nasa_power_soil(...) # <-- Removed
    current_env_df = get_openmeteo_daily_env(lat, lon, start_curr, end_curr) # <-- Now has soil data

    # --- Past Data (One Week Ago) ---
    start_past = one_week_ago - timedelta(days=1)
    end_past = one_week_ago
    past_weather_df = get_openmeteo_data(lat, lon, start_past, end_past)
    # past_soil_df = get_nasa_power_soil(...) # <-- Removed
    past_env_df = get_openmeteo_daily_env(lat, lon, start_past, end_past) # <-- Now has soil data
    
    # --- Return 4 items ---
    return (
        lat, lon,
        current_weather_df, current_env_df,
        past_weather_df, past_env_df
    )

# -----------------------------
# 5️⃣ DATA COLLECTION LOOP (Now Concurrent)
# -----------------------------
def collect_countrywide_data(coords, today, one_week_ago):
    """
    --- SIMPLIFIED ---
    """
    logging.info(f"Starting bulk data collection for {len(coords)} grid points using {MAX_WORKERS} workers...")
    current_weather, past_weather = [], []
    # current_soil, past_soil = [], [] # <-- Removed
    current_env, past_env = [], []
    
    total_coords = len(coords)
    
    # Prepare list of tasks for the thread pool
    tasks = []
    for lat, lon in coords:
        tasks.append((lat, lon, today, one_week_ago))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map tasks to futures
        future_to_point = {executor.submit(fetch_data_for_point, task): task for task in tasks}
        
        logging.info(f"Submitted all {total_coords} tasks to thread pool.")
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_point)):
            point_data = future_to_point[future]
            lat, lon = point_data[0], point_data[1]
            try:
                # --- Unpack 4 items ---
                (
                    lat, lon,
                    curr_w, curr_e,
                    past_w, past_e
                ) = future.result()
                
                # Append data to lists
                current_weather.append(curr_w)
                # current_soil.append(curr_s) # <-- Removed
                current_env.append(curr_e)
                past_weather.append(past_w)
                # past_soil.append(past_s) # <-- Removed
                past_env.append(past_e)
                
                logging.info(f"Processed point {i+1}/{total_coords} at ({lat}, {lon})... COMPLETE.")
                
            except Exception as e:
                logging.error(f"Point ({lat}, {lon}) failed with an unexpected error: {e}", exc_info=True)

    logging.info("...Bulk data collection finished.")
    logging.info("Concatenating all dataframes...")

    # Concatenate, checking for empty lists to avoid errors
    df_current_weather = pd.concat(current_weather, ignore_index=True) if current_weather else pd.DataFrame()
    df_past_weather = pd.concat(past_weather, ignore_index=True) if past_weather else pd.DataFrame()
    # df_current_soil = pd.concat(current_soil, ignore_index=True) if current_soil else pd.DataFrame() # <-- Removed
    # df_past_soil = pd.concat(past_soil, ignore_index=True) if past_soil else pd.DataFrame() # <-- Removed
    df_current_env = pd.concat(current_env, ignore_index=True) if current_env else pd.DataFrame()
    df_past_env = pd.concat(past_env, ignore_index=True) if past_env else pd.DataFrame()

    logging.info("...Concatenation complete.")
    # --- Return 4 items ---
    return df_current_weather, df_past_weather, df_current_env, df_past_env

# -----------------------------
# 6️⃣ MERGE AND ANALYZE
# -----------------------------
def merge_all(weather, env):
    """
    Merges weather and environment dataframes.
    --- SIMPLIFIED ---
    """
    # Ensure all dataframes have the required columns for merging, even if empty
    if weather.empty:
        weather = pd.DataFrame(columns=["timestamp", "lat", "lon", "temperature", "precipitation", "humidity"])
    if env.empty:
        env = pd.DataFrame(columns=["timestamp", "lat", "lon", "et_fao_56", "soil_temp_0_7cm", "soil_moisture_0_to_7cm"])

    # --- FIX: Convert to datetime *after* concat and empty checks. ---
    # This ensures the column is the correct type before using .dt
    # Use errors='coerce' to handle any potential NaNs or bad data from empty DFs
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], errors='coerce')
    env['timestamp'] = pd.to_datetime(env['timestamp'], errors='coerce')

    # Daily data (env) needs to be merged with hourly (weather)
    # We will merge on date, lat, lon. First, get date from timestamp.
    weather['date'] = weather['timestamp'].dt.date
    env['date'] = env['timestamp'].dt.date

    # Aggregate hourly weather data to daily mean
    if not weather.empty:
        # Drop rows where date conversion might have failed
        weather = weather.dropna(subset=['date'])
        daily_weather = weather.groupby(['date', 'lat', 'lon']).agg(
            temperature=('temperature', 'mean'),
            precipitation=('precipitation', 'sum'),
            humidity=('humidity', 'mean')
        ).reset_index()
    else:
        daily_weather = pd.DataFrame(columns=['date', 'lat', 'lon', 'temperature', 'precipitation', 'humidity'])
    # Drop rows where date conversion might have failed in env
    if not env.empty:
        env = env.dropna(subset=['date'])

    # Merge daily datasets
    # df = daily_weather.merge(soil, on=["date", "lat", "lon"], how="outer") # <-- Removed
    df = daily_weather.merge(env, on=["date", "lat", "lon"], how="outer")
    
    # Clean up timestamp columns that came from merges
    df = df.drop(columns=[col for col in df.columns if 'timestamp_' in str(col)]) # Safer check
    df = df.drop(columns=[col for col in df.columns if '_x' in str(col) or '_y' in str(col)]) # Clean up merge artifacts
    return df

# --- Main script execution ---
(
    # --- Unpack 4 items ---
    df_current_weather, df_past_weather,
    df_current_env, df_past_env
) = collect_countrywide_data(coords, today, one_week_ago)

logging.info("Merging current and previous week's data...")
# --- Pass 2 items ---
df_current = merge_all(df_current_weather, df_current_env)
df_previous = merge_all(df_past_weather, df_past_env)

logging.info("Calculating delta features per grid point...")
final_rows = []
# Group by (lat, lon) on the original, merged dataframes
all_coords = set(list(df_current[["lat", "lon"]].itertuples(index=False, name=None)) + 
                 list(df_previous[["lat", "lon"]].itertuples(index=False, name=None)))

for lat, lon in all_coords:
    if pd.isna(lat) or pd.isna(lon):
        continue
        
    curr_df = df_current[(df_current["lat"] == lat) & (df_current["lon"] == lon)]
    prev_df = df_previous[(df_previous["lat"] == lat) & (df_previous["lon"] == lon)]
    
    row = {"lat": lat, "lon": lon}
    # --- Update column list ---
    for col in ["temperature", "precipitation", "humidity", "soil_moisture_0_to_7cm", "et_fao_56", "soil_temp_0_7cm"]:
        curr_mean = curr_df[col].mean() if col in curr_df and not curr_df[col].isnull().all() else np.nan
        prev_mean = prev_df[col].mean() if col in prev_df and not prev_df[col].isnull().all() else np.nan
        
        delta = np.nan
        if pd.notna(curr_mean) and pd.notna(prev_mean):
            delta = curr_mean - prev_mean
        elif pd.notna(curr_mean):
            delta = curr_mean # No previous data, so delta is just the current value
        
        row[f"{col}_current_mean"] = curr_mean
        row[f"{col}_prev_mean"] = prev_mean
        row[f"{col}_delta"] = delta
        row[f"{col}_change_per_day"] = delta / days_window if pd.notna(delta) else np.nan
        
    final_rows.append(row)

final_df = pd.DataFrame(final_rows)
logging.info("...Delta analysis complete.")

# -----------------------------
# 7️⃣ HISTORICAL LANDSLIDES
# -----------------------------
landslides = get_landslide_data_india()
landslides_recent = pd.DataFrame()
if not landslides.empty:
    landslides_recent = landslides[landslides["date"] >= pd.to_datetime(one_week_ago)]

# -----------------------------
# 8️⃣ SHOW FINAL OUTPUT & SAVE
# -----------------------------
logging.info("\n✅ Current Week Data Sample (daily aggregated):\n" + df_current.head().to_string())
logging.info("\n✅ Previous Week Data Sample (daily aggregated):\n" + df_previous.head().to_string())
logging.info("\n✅ Final Delta DataFrame (per-point):\n" + final_df.head().to_string())
logging.info("\n✅ Recent Landslides in India:\n" + landslides_recent.head().to_string())

logging.info("Saving all data to CSV files...")
try:
    df_current.to_csv("current_week_data_grid.csv", index=False)
    df_previous.to_csv("previous_week_data_grid.csv", index=False)
    final_df.to_csv("delta_features_grid.csv", index=False)
    landslides_recent.to_csv("recent_landslides.csv", index=False)
    logging.info("...Successfully saved all CSV files.")
except Exception as e:
    logging.error(f"Failed to save CSV files: {e}")

logging.info("-------------------------------------------------")
logging.info("✅ Script Finished")
logging.info("-------------------------------------------------")

