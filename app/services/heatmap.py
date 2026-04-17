import cv2
import numpy as np
import os

def generate_heatmap(image: np.ndarray, mask: np.ndarray, output_path: str = "outputs/heatmap.png") -> str:
    """
    Generate a heatmap based on ExG value classification.
    High -> Green, Medium -> Yellow, Low -> Red.
    Overlays the heatmap on the original image.
    """
    b, g, r = cv2.split(image)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    exg = 2 * g - r - b

    # Create empty heatmap
    heatmap = np.zeros_like(image)

    # Map colors
    # High (Healthy) > 50 -> Green BGR(0, 255, 0)
    heatmap[exg > 50] = [0, 255, 0]
    # Medium (Mild) 20-50 -> Yellow BGR(0, 255, 255)
    heatmap[(exg <= 50) & (exg >= 20)] = [0, 255, 255]
    # Low (Severe) < 20 -> Red BGR(0, 0, 255)
    heatmap[exg < 20] = [0, 0, 255]

    # Mask out background
    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

    # Overlay on original image
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)

    return output_path
