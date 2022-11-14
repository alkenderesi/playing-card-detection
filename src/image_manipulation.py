import cv2
import numpy as np


RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A')
SUITS = ('spades', 'hearts', 'diamonds', 'clubs')


def create_white_mask(gray):
    """
    Creates mask from grayscale image filtering white parts.
    """

    # Maximum pixel intensity
    max_intensity = np.max(gray)

    # Mask from high intensity pixels
    white_mask = cv2.inRange(gray, int(max_intensity * 0.75), int(max_intensity))

    return white_mask


def find_contour_corners(contour):
    """
    Finds 4 corner points of a contour.
    """

    # Approximate contour polygon points
    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, perimeter * 0.02, True)

    # Not quadrilateral
    if len(corners) != 4:
        return None

    # Order correction
    if np.linalg.norm(corners[0] - corners[1]) < np.linalg.norm(corners[0] - corners[-1]):
        corners = np.roll(corners, -1, axis=0)

    return corners


def create_birds_eye_view(img, points, output_shape):
    """
    Transforms and crops an image to create a birds eye view.
    """

    # Output points for transformation
    output_points = [
        [0, 0], # Top left
        [0, output_shape[1]], # Bottom left
        [output_shape[0], output_shape[1]], # Bottom right
        [output_shape[0], 0], # Top right
    ]

    # Calculate perspective transform
    transform = cv2.getPerspectiveTransform(
        np.float32(points),
        np.float32(output_points)
    )

    # Apply perspective transform and crop
    birds_eye = cv2.warpPerspective(img, transform, output_shape)

    return birds_eye


def extract_card_properties(img):
    """
    Extracts rank and suit from the top left corner of a card image.
    """

    # Edge detection
    edges = cv2.Canny(img, 100, 200)

    # Binary conversion
    binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]

    # Crop card rank and suit
    rank_crop = binary[0:54, 0:34] # 34x54
    suit_crop = binary[54:88, 0:34] # 34x34

    return rank_crop, suit_crop


def create_card_label(img, contour, corners, rank, suit):
    """
    Labels and colors a segmented card.
    """

    # Mix color
    color = (int((rank + 1) / len(RANKS) * 255), int((suit + 1) / len(SUITS) * 255))

    # Draw contour
    cv2.drawContours(img, [contour], -1, color, 3)

    # Blank mask
    mask = np.zeros_like(img)

    # Fill contour on mask
    cv2.fillPoly(mask, [contour], color)

    # Add mask to image
    img = cv2.addWeighted(img, 0.95, mask, 0.70, 0)

    # Label text
    label = RANKS[rank] + ' ' + SUITS[suit]

    # Draw label
    cv2.putText(img, label, corners[0][0], cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

    return img
