import cv2
import numpy as np
import os
import sys

# Ensure this accesses the local app module when run from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.preprocessing import white_balance, segment_leaf
from app.services.features import extract_rgb_means, compute_indices
from app.services.analysis import compute_deficiency, calculate_deficiency_percentage, final_classification
from app.services.heatmap import generate_heatmap

def run_test(image_path: str):
    print(f"==================================================")
    print(f"🧪 Running Pipeline on: {image_path}")
    print(f"==================================================")
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return

    # Load Image
    image = cv2.imread(image_path)
    
    # 1. Preprocess
    print("-> Preprocessing: White balancing & Segmenting...")
    wb_image = white_balance(image)
    segmented_img, mask = segment_leaf(wb_image)
    
    # Check if proper segmentation occurred
    if cv2.countNonZero(mask) == 0:
        print("-> Result: No leaf tissue found (empty mask)")
        return
        
    # 2. Extract Features
    print("-> Extracting features...")
    rgb_means = extract_rgb_means(segmented_img, mask)
    print(f"   Means - R: {rgb_means['R']:.2f}, G: {rgb_means['G']:.2f}, B: {rgb_means['B']:.2f}")
    
    indices = compute_indices(rgb_means["R"], rgb_means["G"], rgb_means["B"])
    print(f"   Indices - ExG: {indices['ExG']:.2f}, NGI: {indices['NGI']:.2f}, GLI: {indices['GLI']:.2f}")
    
    # 3. Analyze Deficiency
    print("-> Analyzing deficiency...")
    def_data = compute_deficiency(segmented_img, mask)
    total_pixels = def_data["healthy_pixels"] + def_data["deficient_pixels"]
    print(f"   Pixels - Healthy: {def_data['healthy_pixels']}, Deficient: {def_data['deficient_pixels']}")
    
    percent_def = calculate_deficiency_percentage(def_data["deficient_pixels"], total_pixels)
    status = final_classification(percent_def)
    
    print(f"-> Result: {percent_def:.2f}% Deficiency -> {status} Status")
    
    # 4. Generate Heatmap
    print("-> Generating heatmap...")
    heatmap_path = f"outputs/test_heatmap_{os.path.basename(image_path)}"
    generate_heatmap(segmented_img, mask, heatmap_path)
    print(f"-> Heatmap saved at: {heatmap_path}")
    print(f"==================================================")

if __name__ == "__main__":
    # Create output dir if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    dummy_image_path = "data/dummy_test_leaf.png"
    
    if not os.path.exists(dummy_image_path):
        # Create a dummy greenish image representing a segmented leaf structure
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Background: noisy dark
        # Target: Green Leaf (Healthy region)
        cv2.circle(img, (100, 100), 50, (30, 200, 30), -1) 
        # Target: Deficient region (yellow/red spot)
        cv2.circle(img, (120, 120), 10, (30, 100, 150), -1) 
        cv2.imwrite(dummy_image_path, img)
        print(f"Created a dummy testing leaf image at {dummy_image_path}")
        
    run_test(dummy_image_path)
