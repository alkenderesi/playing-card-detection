import os
import sys
import cv2


device = int(sys.argv[1])
res_x = int(sys.argv[2])
res_y = int(sys.argv[3])
output_path = sys.argv[4]

capture = cv2.VideoCapture(device)
capture.set(3, res_x)
capture.set(4, res_y)
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

if output_path and not os.path.exists(output_path):
    os.makedirs(output_path)

for i in range(320):
    frame = capture.read()[1]
    cv2.imshow('Frame', frame)
    cv2.imwrite(f'{output_path}/{i}.jpg', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break

capture.release()
cv2.destroyAllWindows()