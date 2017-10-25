import time
import numpy
import cv2
import image_conversion as ic
import threading as th

#Start playing live video 
cap = cv2.VideoCapture(0)

cap.set(3, 200)
cap.set(4, 200)

retval, frame = cap.read()

height, width, bpp = numpy.shape(frame)
start = 0

#Pixel array to store 1's and -1's
image = [[0] * width for i in range(height)]
while(True):
        #Captures individual frame from video
        ret, frame = cap.read()
        
        #Grayscales image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Turns pixels to either black or white
        returnVal, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        #Height, width and color channel of image (color channel not important)
        height, width, bpp = numpy.shape(frame)

        #Initializes image array with 1's and -1's representing white and blackrespectively
        start = time.time()
        for y in range(0,height):
            for x in range(0, width):
                image[y][x] = 0 if thresh[y][x] == 0 else 1
        #Shows live video
        cv2.imshow('frame', thresh)

        #Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break;
cap.release()
cv2.destroyAllWindows()
test = ic.convert_img_to_bw_hand_50x50(image)
end = time.time()
print(end-start)
for a in test:
    print(a)
