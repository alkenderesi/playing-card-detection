import os
import sys
import cv2


device = int(sys.argv[1])
res_x = int(sys.argv[2])
res_y = int(sys.argv[3])
sample_count = int(sys.argv[4])
output_path = sys.argv[5]

capture = cv2.VideoCapture(device)
capture.set(3, res_x)
capture.set(4, res_y)
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

if output_path and not os.path.exists(output_path):
    os.makedirs(output_path)

for i in range(sample_count):
    frame = capture.read()[1]
    cv2.imshow('Frame', frame)
    cv2.imwrite('{0}/{1}.jpg'.format(output_path, i), frame)
    if i % (sample_count / 10) == 0:
        print('{0}%'.format(i / sample_count * 100))
    key = cv2.waitKey(1)
    if key != -1:
        break

capture.release()
cv2.destroyAllWindows()
