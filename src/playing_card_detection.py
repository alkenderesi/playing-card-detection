import os
import numpy as np
import cv2
from .image_manipulation import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable TensorFlow warnings
import tensorflow as tf

RANK_MODEL = tf.keras.models.load_model('models/rank/rank_model')
SUIT_MODEL = tf.keras.models.load_model('models/suit/suit_model')


def detect_playing_cards(img):
    """
    Detect playing cards in an image.
    """

    # Gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mask white parts
    white_mask = create_white_mask(gray)

    # Find contours
    contours = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Iterate over contours
    for contour in contours:

        # Filter contours by area
        if cv2.contourArea(contour) < img.shape[0] * img.shape[1] * 0.025:
            continue

        # Find corners
        corners = find_contour_corners(contour)

        # Filter contours by corners
        if corners is None:
            continue

        # Invert colors
        inverted = 255 - gray

        # Birds eye view transformation
        birds_eye = create_birds_eye_view(inverted, corners, (500, 700))

        # Crop rank and suit sections
        rank_crop = birds_eye[0:110, 0:60] # 60x110
        suit_crop = birds_eye[110:180, 0:60] # 60x70

        # Reshape crops for the models
        rank_crop = rank_crop.reshape(1, 110, 60)
        suit_crop = suit_crop.reshape(1, 70, 60)

        # Predict rank and suit with neural networks
        rank_pred = RANK_MODEL.predict(rank_crop, verbose=0)
        suit_pred = SUIT_MODEL.predict(suit_crop, verbose=0)

        # Get ids and confidences
        rank = np.argmax(rank_pred)
        suit = np.argmax(suit_pred)
        rank_conf = np.around(np.max(rank_pred), 2)
        suit_conf = np.around(np.max(suit_pred), 2)

        # Label the detected card
        img = create_card_label(img, contour, corners, rank, rank_conf, suit, suit_conf)
    
    return img
