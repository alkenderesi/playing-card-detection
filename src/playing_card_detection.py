import cv2
import numpy as np
from . import image_manipulation


def detect_playing_cards(img):
    """
    Detect playing cards in an image.
    *Temporarily used for testing basic functionality*
    """

    # Mask white parts
    white_mask = image_manipulation.create_white_mask(img)

    # Find contours
    contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum area of a contour to be considered a card
    min_area = img.shape[0] * img.shape[1] // 16

    # Blank card mask
    card_mask = np.zeros_like(white_mask)

    # Draw and fill contours
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(card_mask, contour, -1, (255, 255, 255), 1)
            cv2.fillPoly(card_mask, [contour], (255, 255, 255))
    
    # Mask the original image
    cards = cv2.bitwise_and(img, img, mask=card_mask)

    return cards
