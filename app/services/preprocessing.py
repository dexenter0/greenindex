import cv2
import numpy as np

def white_balance(image: np.ndarray) -> np.ndarray:
    """
    White balance the given RGB image.
    This method takes a simple approach by assuming the brightest 
    pixels should be white, or using the grey world assumption.
    Using white patch algorithm for simplicity.
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def segment_leaf(image: np.ndarray) -> np.ndarray:
    """
    Segment the leaf from the background using HSB threshold.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a broader range to include yellowing and chlorotic leaf regions 
    # (Hue: 10 to 100 includes brown, yellow, and green)
    lower_bound = np.array([10, 20, 20])
    upper_bound = np.array([100, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # morphological operations to clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented, mask
