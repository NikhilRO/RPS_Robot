import cv2
import numpy as np
import image_conversion as ic

cap = cv2.VideoCapture(0)

cap.set(3, 160)
cap.set(4, 120)

prev = 0
edges = 0
counter = 0
imgCounter = 1310

while(True):

    retval, frame = cap.read()

    edges = cv2.Canny(frame, 120, 130)

    height, width = np.shape(edges)

    cv2.namedWindow('edge detect', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('edge detect', 640, 480)

    cv2.imshow('edge detect', edges)

    counter += 1

    if (counter % 9 == 0):
        cv2.imwrite('scissors/image'+str(imgCounter)+'.png',edges)
        imgCounter += 1

    '''for y in range(0,height):
            for x in range(0, width):
                image[y][x] = 0 if edges[y][x] < 128 else 1'''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
#arr = ic.to_black_white(ic.compress_to_25x25(image))
'''for a in arr:
    for b in a:
        print(b, end='')
    print()'''
