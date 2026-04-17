from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import os

from app.services.preprocessing import white_balance, segment_leaf
from app.services.features import extract_rgb_means, compute_indices
from app.services.analysis import compute_deficiency, calculate_deficiency_percentage, final_classification
from app.services.heatmap import generate_heatmap
from app.models.schemas import AnalyzeResponse

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image content.")

    # 1. Preprocess
    wb_image = white_balance(image)
    segmented_img, mask = segment_leaf(wb_image)
    
    # 2. Extract Features
    rgb_means = extract_rgb_means(segmented_img, mask)
    indices = compute_indices(rgb_means["R"], rgb_means["G"], rgb_means["B"])
    
    # 3. Analyze Deficiency
    def_data = compute_deficiency(segmented_img, mask)
    
    if def_data["healthy_pixels"] + def_data["deficient_pixels"] == 0:
        raise HTTPException(status_code=400, detail="No leaf tissue detected in image.")
        
    total_pixels = def_data["healthy_pixels"] + def_data["deficient_pixels"]
    percent_def = calculate_deficiency_percentage(def_data["deficient_pixels"], total_pixels)
    status = final_classification(percent_def)
    
    # 4. Generate Heatmap
    heatmap_filename = f"heatmap_{file.filename}.png"
    heatmap_path = os.path.join("outputs", heatmap_filename)
    generate_heatmap(segmented_img, mask, heatmap_path)

    return AnalyzeResponse(
        deficiency_percentage=percent_def,
        status=status,
        indices=indices,
        heatmap_path=heatmap_path
    )
