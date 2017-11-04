import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(3, 200)
cap.set(4, 200)

prev = 0
a = 0
maxi = 0
edges = 0

while(True):

    retval, frame = cap.read()

    edges = cv2.Canny(frame, 100, 100)

    height, width = np.shape(edges)

    cv2.imshow('edge detect', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;


for i in range(0, height):  
    for j in range(0, width):
        if (edges[i][j] == 255 & prev == 0):
            a += 1
            prev = 1
        elif edges[i][j] != 255:
            prev = 0
    if (a > maxi): 
        maxi = a
        a=0
if maxi < 1100:
    print("Scissors")
elif maxi < 1200:
    print("Rock")
else:
    print("Paper")
print(maxi)
cap.release()
cv2.destroyAllWindows()
