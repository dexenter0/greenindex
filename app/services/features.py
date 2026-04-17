import cv2
import numpy as np

def extract_rgb_means(image: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    Given a leaf image, extract the mean RGB values 
    for the leaf pixels. BGR to RGB channel mapping needed.
    """
    if mask is None:
        B, G, R = cv2.split(image)
        mask = np.ones_like(G)
    else:
        # Get only the active pixels
        B = image[:, :, 0][mask > 0]
        G = image[:, :, 1][mask > 0]
        R = image[:, :, 2][mask > 0]

    return {
        "R": np.mean(R),
        "G": np.mean(G),
        "B": np.mean(B)
    }

def compute_indices(R: float, G: float, B: float) -> dict:
    """
    Compute vegetation indices based on R, G, B components.
    Formulas:
    ExG = 2G - R - B
    NGI = G / (R + G + B + 1e-6)
    GLI = (2G - R - B) / (2G + R + B + 1e-6)
    """
    # handle divide by zero
    den = R + G + B + 1e-6
    den_gli = 2 * G + R + B + 1e-6
    
    exg = 2 * G - R - B
    ngi = G / den
    gli = (2 * G - R - B) / den_gli
    
    return {
        "ExG": exg,
        "NGI": ngi,
        "GLI": gli
    }
