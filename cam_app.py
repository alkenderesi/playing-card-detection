import sys
import cv2
from src.playing_card_detection import detect_playing_cards


def main():

    device = int(sys.argv[1])
    res_x = int(sys.argv[2])
    res_y = int(sys.argv[3])

    capture = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    capture.set(3, res_x)
    capture.set(4, res_y)

    _, test_frame = capture.read()
    if test_frame.shape[0] != res_y or test_frame.shape[1] != res_x:
        print(f'Unsupported resolution by webcam: {res_x}x{res_y}', file=sys.stderr)
        capture.release()
        return

    while True:
        _, frame = capture.read()
        cards = detect_playing_cards(frame)
        cv2.imshow('Cards', cards)
        key = cv2.waitKey(1)
        if key != -1:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
