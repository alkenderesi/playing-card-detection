import os
import sys
import cv2
from src.playing_card_detection import detect_playing_cards


def main():

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)

    _, _, input_images = next(os.walk(input_path))

    for image_name in input_images:
        img = cv2.imread(os.path.join(input_path, image_name))
        cards = detect_playing_cards(img)
        if output_path:
            cv2.imwrite(os.path.join(output_path, image_name), cards)
        else:
            cv2.imshow(image_name, cards)
            key = cv2.waitKey(0)
            cv2.destroyWindow(image_name)
            if key == 27:
                break


if __name__ == '__main__':
    main()
