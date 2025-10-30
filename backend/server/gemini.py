import uvicorn
import httpx
import base64
import os
import logging
from dotenv import load_dotenv
# import io  <-- No longer needed
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
# from PIL import Image, ExifTags  <-- No longer needed

# -----------------
# 1. SETUP LOGGING
# -----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -----------------
# 2. SYSTEM PROMPT FOR GEMINI (Reverted)
# -----------------
SYSTEM_PROMPT = """
You are an expert geospatial analyst specializing in landslide risk assessment.
Your task is to analyze an aerial (drone) image of a terrain and identify
potential landslide risks.

Based *only* on the visual evidence in the image, you will perform three tasks:

1.  **Write `analysis_summary`**: Write a brief, one-sentence summary of the
    primary risk factors you can see (e.g., "Steep, unconsolidated soil with
    visible water saturation and sparse vegetation.").

2.  **Estimate Metadata**:
    * `estimated_location`: Guess the general region based on terrain,
        vegetation, and building styles (e.g., "Himalayan foothills, India"
        or "Coastal cliffs, unknown region").
    * `estimated_timestamp`: Guess the time of day and season based on
        lighting, shadows, and vegetation state (e.g., "Mid-day, post-monsoon season").

3.  **Identify `risk_heatmap`**: Pinpoint 3-5 specific areas of high, medium, or
    low risk.
    * Provide *estimated* latitude and longitude for each point. Since you
        don't have real GPS, use a relative local coordinate system, assuming
        the image center is `(0.0, 0.0)`. For example, a point to the top-left
        could be `(0.0001, -0.0001)`.
    * Provide a `risk_level` from 1 (Low) to 5 (High).
    * Provide a `description` justifying why it's a risk
        (e.g., "Visible soil creep and cracks at top of slope").

You MUST return your response ONLY in the specified JSON format.
"""

# -----------------
# 3. PYDANTIC MODELS (Reverted)
# -----------------
class RiskPoint(BaseModel):
    # --- Reverted: Lat/Lon are now relative estimates ---
    latitude: float = Field(..., description="Estimated relative latitude (center is 0.0)")
    longitude: float = Field(..., description="Estimated relative longitude (center is 0.0)")
    risk_level: int = Field(..., ge=1, le=5, description="Risk on a scale of 1 (Low) to 5 (High)")
    description: str = Field(..., description="Brief justification for the risk level")

class ImageMetadata(BaseModel):
    # --- Reverted: This data now comes from Gemini ---
    estimated_location: str = Field(..., description="Gemini's estimate of the location")
    estimated_timestamp: str = Field(..., description="Gemini's estimate of the time/season")
    analysis_summary: str = Field(..., description="Gemini's one-sentence summary of the main risk factor")

# This is the final JSON object the API will return
class AnalysisResponse(BaseModel):
    metadata: ImageMetadata
    risk_heatmap: List[RiskPoint]

# --- Reverted: This is now the model for what GEMINI returns ---
class GeminiAnalysis(BaseModel):
    estimated_location: str = Field(..., description="Gemini's estimate of the location")
    estimated_timestamp: str = Field(..., description="Gemini's estimate of the time/season")
    analysis_summary: str = Field(..., description="One-sentence summary of the main risk factor")
    risk_heatmap: List[RiskPoint]

# -----------------
# 4. FASTAPI APP SETUP
# -----------------
app = FastAPI(
    title="Landslide Risk Analysis API",
    description="Upload an aerial image and get a JSON response with metadata and risk heatmap coordinates.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------
# 5. EXIF HELPER FUNCTIONS (REMOVED)
# -----------------
# ... (All EXIF functions removed) ...

# -----------------
# 6. GEMINI API HELPER FUNCTION (Reverted)
# -----------------
async def call_gemini_vision_api(image_bytes: bytes) -> GeminiAnalysis:
    """
    Calls the Gemini API with just the image, returns a structured JSON response.
    """
    load_dotenv()
    api_key = (
        os.getenv("GEMINI_API")
      )  # This will be injected by the environment
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    # Encode the image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # --- NEW, ROBUST FIX for 400 Bad Request ---
    # The API doesn't support $defs, definitions, or $ref.
    # We must manually inline the schema for `RiskPoint` into `GeminiAnalysis`.

    log.info("Generating and flattening JSON schema for API...")
    
    # 1. Get the schema for the sub-model (RiskPoint)
    risk_point_schema = RiskPoint.model_json_schema()
    # Pydantic v2 might nest it inside "$defs"
    if "$defs" in risk_point_schema and "RiskPoint" in risk_point_schema["$defs"]:
        risk_point_def = risk_point_schema["$defs"]["RiskPoint"]
    else:
        risk_point_def = risk_point_schema
    # Remove title if it exists, as API might not like it
    risk_point_def.pop("title", None)

    # 2. Get the schema for the main model (GeminiAnalysis)
    schema = GeminiAnalysis.model_json_schema()

    # 3. Find the 'risk_heatmap' property (which is an array) and 
    #    find its 'items' (which points to the sub-model)
    if "properties" in schema and "risk_heatmap" in schema["properties"]:
        if "items" in schema["properties"]["risk_heatmap"]:
            # 4. Replace the 'items' (which was a $ref) with the
            #    full, inlined schema of RiskPoint
            schema["properties"]["risk_heatmap"]["items"] = risk_point_def
            log.info("Successfully inlined RiskPoint schema into risk_heatmap items.")
        else:
            log.error("Could not find 'items' in 'risk_heatmap' schema.")
    else:
        log.error("Could not find 'properties' or 'risk_heatmap' in main schema.")

    # 5. Remove the top-level "$defs" or "definitions" block entirely,
    #    as the API rejects it.
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    
    # --- END FIX ---
    
    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    # --- Reverted: Simplified prompt, no EXIF data ---
                    {"text": "Analyze this aerial image for landslide risk."},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.2
        }
    }
    
    # Use httpx for an asynchronous API call
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            log.info("Sending request to Gemini API (image-only)...")
            response = await client.post(api_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status() 
            result = response.json()
            
            json_text = result.get("candidates", [{}])[0] \
                              .get("content", {}) \
                              .get("parts", [{}])[0] \
                              .get("text", "{}")
            
            log.info("Successfully received and parsed response from Gemini.")
            
            parsed_response = GeminiAnalysis.model_validate_json(json_text)
            return parsed_response
        
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {e.response.text}")
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# -----------------
# 7. API ENDPOINT (Reverted)
# -----------------
@app.post("/analyze/", 
          response_model=AnalysisResponse,
          summary="Analyze Aerial Image for Landslide Risk")
async def analyze_image_endpoint(
    file: UploadFile = File(..., description="The aerial image file (JPEG, PNG) to analyze.")
):
    """
    Upload an image of a terrain to receive a structured JSON analysis of:
    1.  **Metadata**: AI-estimated location, time, and summary.
    2.  **Risk Heatmap**: A list of coordinates with specific risk levels (1-5)
        and descriptions, perfect for plotting on a 3D model.
    """
    log.info(f"Received file: {file.filename} (Content-Type: {file.content_type})")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (JPEG, PNG).")
        
    image_bytes = await file.read()
    
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image file is too large (max 15MB).")

    # --- Reverted: No EXIF extraction ---
    
    # --- Call Gemini with only the image ---
    gemini_result = await call_gemini_vision_api(image_bytes)
    
    # --- Reverted: Construct the final response from Gemini's output ---
    
    final_metadata = ImageMetadata(
        estimated_location=gemini_result.estimated_location,
        estimated_timestamp=gemini_result.estimated_timestamp,
        analysis_summary=gemini_result.analysis_summary
    )
    
    final_response = AnalysisResponse(
        metadata=final_metadata,
        risk_heatmap=gemini_result.risk_heatmap
    )
    
    return final_response

# -----------------
# 8. RUN THE APP
# -----------------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    log.info(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

