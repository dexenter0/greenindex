import cv2
import numpy as np

def classify_pixel(exg: float) -> str:
    """
    Classify a single pixel's ExG value.
    Note: threshold depends on the sensor range. Assuming 8-bit scale [-510, 510].
    Generally, higher ExG suggests more greenness.
    Arbitrary boundaries chosen based on typical ExG for plants:
    Low: < 20
    Medium: 20-50
    High: > 50
    """
    if exg > 50:
        return "healthy"
    elif exg >= 20:
        return "mild"
    else:
        return "severe"

def compute_deficiency(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute ExG for each pixel in the segmented leaf area.
    Then count healthy vs deficient pixels based on the ExG values.
    Returns counts.
    """
    b, g, r = cv2.split(image)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    exg = 2 * g - r - b

    # only consider leaf pixels based on mask
    valid_exg = exg[mask > 0]
    total_pixels = len(valid_exg)

    # Calculate deficiency directly
    # Using threshold <= 50 as deficient (including mild & severe)
    deficient_count = np.sum(valid_exg <= 50)
    
    return {
        "healthy_pixels": total_pixels - deficient_count,
        "deficient_pixels": deficient_count
    }

def calculate_deficiency_percentage(deficient_pixels: int, total_pixels: int) -> float:
    """
    Calculate percentage of deficient pixels.
    deficient_pixels / total_pixels * 100
    """
    if total_pixels == 0:
        return 0.0
    return (deficient_pixels / total_pixels) * 100.0

def final_classification(percentage: float) -> str:
    """
    Classify based on deficient pixels percentage.
    0-20 -> Healthy
    20-50 -> Mild
    50+ -> Severe
    """
    if percentage > 50:
        return "Severe"
    elif percentage >= 20:
        return "Mild"
    else:
        return "Healthy"
