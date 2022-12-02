import sys
import cv2
from src.playing_card_detection import detect_playing_cards


def main():

    device = int(sys.argv[1])
    res_x = int(sys.argv[2])
    res_y = int(sys.argv[3])

    capture = cv2.VideoCapture(device)
    capture.set(3, res_x)
    capture.set(4, res_y)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    test_frame = capture.read()[1]
    if test_frame.shape[0] != res_y or test_frame.shape[1] != res_x:
        print('Unsupported resolution by webcam: {0}x{1}'.format(res_x, res_y), file=sys.stderr)
        capture.release()
        return

    while True:
        frame = capture.read()[1]
        cards = detect_playing_cards(frame)
        cv2.imshow('Cards', cards)
        key = cv2.waitKey(1)
        if key != -1:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
