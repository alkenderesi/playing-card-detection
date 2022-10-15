import cv2
import numpy as np


def create_white_mask(img):
    """
    Creates mask from BGR image filtering white parts.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Maximum pixel intensity
    max_intensity = np.max(gray)

    # Mask from high intensity pixels
    white_mask = cv2.inRange(gray, int(max_intensity * 0.75), int(max_intensity))

    return white_mask
