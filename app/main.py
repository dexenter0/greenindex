from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes.analyze import router as analyze_router
import os

app = FastAPI(
    title="GreenIndex - Chlorophyll Deficiency Detection System",
    description="Analyze plant leaves from images to extract vegetation indices, determine health status, and generate heatmaps.",
    version="1.0.0"
)

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Mount outputs for static file serving (heatmap images)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.include_router(analyze_router)

@app.get("/", response_class=FileResponse)
def serve_ui():
    """Serve the Web UI for uploading images."""
    return FileResponse("app/templates/index.html")
