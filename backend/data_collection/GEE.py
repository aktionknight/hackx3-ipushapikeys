import ee
import pandas as pd
import logging
import time

# ----------
# 1. SETUP
# ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

try:
    # --- FIX ---
    # Add your new, clean Project ID here.
    # This forces GEE to use your new project, bypassing the blocked "walmart" one.
    NEW_PROJECT_ID = 'hackx-476713' # <-- PASTE YOUR NEW PROJECT ID HERE
    
    # Initialize the Earth Engine library.
    ee.Initialize(project=NEW_PROJECT_ID)
    logging.info(f"✅ Google Earth Engine API initialized successfully with project: {NEW_PROJECT_ID}.")
except ee.ee_exception.EEException:
    logging.error("❌ Failed to initialize Earth Engine.")
    logging.error("Please run 'earthengine authenticate' in your terminal.")
    exit()
except Exception as e:
    logging.error(f"❌ An unexpected error occurred: {e}")
    exit()

# ----------
# 2. DEFINE DATASETS
# ----------
# Define the GEE Image/Collections we will sample
# 1. Elevation & Slope
srtm = ee.Image("USGS/SRTMGL1_003")
elevation = srtm.select('elevation')
# Calculate slope from the elevation data
slope = ee.Terrain.slope(elevation)

# 2. Soil Type (Using SoilGrids)
soil_type = ee.Image("projects/soilgrids-isric/soc_mean") # Example: Soil Organic Carbon

# 3. Landcover (Using Copernicus)
landcover = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019") \
            .select('discrete_classification')

# 4. Distance to River (Using Global Surface Water)
# Create an image where 1 = permanent water, 0 = not
water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
# Set a threshold for "permanent" water (e.g., > 50% occurrence)
water_mask = water.gt(50)
# Create a distance-to-water image (in meters)
# Max distance set to 10km (10000m) for efficiency
distance_to_water = water_mask.fastDistanceTransform().sqrt()

logging.info("🌍 GEE static datasets loaded (Elevation, Slope, Soil, Landcover, Water Distance).")

# ----------
# 3. HELPER FUNCTION
# ----------
def get_static_features_for_point(lat, lon):
    """
    Samples all static datasets for a single (lat, lon) point.
    """
    try:
        # Create a GEE Point object
        point = ee.Geometry.Point(lon, lat)
        
        # 1. Sample Elevation
        elev_data = elevation.sample(point, 30).first().get('elevation').getInfo()
        
        # 2. Sample Slope
        slope_data = slope.sample(point, 30).first().get('slope').getInfo()
        
        # 3. Sample Soil Type
        # SoilGrids has many layers, we'll sample 'soc_mean' (Soil Organic Carbon)
        soil_data = soil_type.sample(point, 250).first().get('soc_mean').getInfo()
        
        # 4. Sample Landcover
        # This returns a class number (e.g., 20, 30, 40)
        lc_data = landcover.sample(point, 100).first().get('discrete_classification').getInfo()
        
        # 5. Sample Distance to Water
        dist_water_data = distance_to_water.sample(point, 30).first().get('distance').getInfo()

        return {
            "elevation": elev_data,
            "slope": slope_data,
            "soil_type": soil_data,
            "landcover": lc_data,
            "distance_to_river": dist_water_data
        }
    except Exception as e:
        # GEE often returns None for points with no data (e.g., ocean)
        # logging.warning(f"Could not get data for ({lat}, {lon}): {e}")
        return {
            "elevation": None,
            "slope": None,
            "soil_type": None,
            "landcover": None,
            "distance_to_river": None
        }

# ----------
# 4. MAIN EXECUTION
# ----------
def enrich_dataset(input_csv, output_csv):
    """
    Loads the input CSV, enriches it with static features, and saves a new CSV.
    """
    logging.info(f"Loading input file: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"❌ ERROR: Input file not found: {input_csv}")
        logging.error("Please run 'landslide_data_fetcher_with_logging.py' first.")
        return

    if df.empty:
        logging.error("❌ ERROR: Input file is empty.")
        return

    logging.info(f"Found {len(df)} points to enrich.")
    
    static_features = []
    total = len(df)
    for i, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        if pd.isna(lat) or pd.isna(lon):
            continue
            
        logging.info(f"Processing point {i+1}/{total} at ({lat}, {lon})...")
        features = get_static_features_for_point(lat, lon)
        features['lat'] = lat
        features['lon'] = lon
        static_features.append(features)
        
        # GEE has its own rate limits, so a small sleep is wise
        time.sleep(0.1) 

    logging.info("...Enrichment complete.")
    
    # Merge the new static features back into the original dataframe
    static_df = pd.DataFrame(static_features)
    
    # Use 'lat' and 'lon' as keys to merge
    final_df = pd.merge(df, static_df, on=['lat', 'lon'], how='left')
    
    logging.info(f"Saving enriched dataset to {output_csv}")
    final_df.to_csv(output_csv, index=False)
    
    logging.info("\n✅ Final Enriched Data Sample:\n" + final_df.head().to_string())
    logging.info("✅ Script Finished.")

if __name__ == "__main__":
    INPUT_FILE = "delta_features_grid.csv"
    OUTPUT_FILE = "final_enriched_dataset.csv"
    enrich_dataset(INPUT_FILE, OUTPUT_FILE)

