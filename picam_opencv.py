from picamera.array import piRGBArray
from picamera import PiCamera
import time
import cv2

#Init the camera and grab reference to raw camera capture
cam = PiCamera()
cam.resoution = (640, 480)
cam.framerate = 32
rawCap = PiRGBArray(cam, size=(640, 480))

#Give camera time to warm up
time.sleep(1)

#Capture frames from camera using infinite loop
for frame in cam.capture_continuous(rawCap, format = "bgr", use_video_port=True):
    #Grab NumPy array representing image and store it in the 'image' variable
    image = frame.array

    #Show frame
    cv2.imshow("Frame", image)
    
    #Clear video stream to prepare for next frame
    rawCap.truncate(0)

    #Exit live video when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
