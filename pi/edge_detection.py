from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import image_conversion as ic
import neural_net as nn
import rps_net as rn
import net

c_p = False
c_t = time.time()
cntr = 1
cam = PiCamera()

cam.resolution = (128, 128)
cam.framerate = 40
rawCapture = PiRGBArray(cam, size=(128, 128))

#Camera warm-up time
time.sleep(2)

prev = 0
edges = 0

cam.capture('frame.png')

f = cv2.imread('frame.png', 0)

edges = cv2.Canny(f, 120, 130)

height, width = np.shape(edges)

ROCK = 0
PAPER = 1
SCISSORS = 2

image = [[0 for x in range(width)] for y in range(height)]
net = nn.new_net([0])
w, b = rn.read_w_b_from_file('weights12.txt')
nn.set_network_weights_biases(net, w, b)
for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    arr = []

    if cntr % 4 == 0:
        time.sleep(0.25)

    edges = cv2.Canny(frame.array, 120, 130)

    cv2.namedWindow('edge detect', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('edge detect', 640, 480)

    cv2.imshow('edge detect', edges)

    image = edges/255
    
    image1, is_play = ic.compress_to_25x25(image)
    
    if is_play:
        if not c_p:
            cntr += 1
    c_p = is_play
    
    arr = image1.ravel()
    answer = ROCK
    if cntr%4 == 0:
        print("ayylmao")
        answer = nn.get_output(net, arr)
        cntr=1

        if answer == ROCK:
            print("Rock", cntr)
        elif answer == PAPER:
            print("Paper", cntr)
        elif answer == SCISSORS:
            print("Scissors", cntr)
    elif not cntr == 1:
        if time.time() - c_t > 4:
            cntr = 1
            c_t = time.time()
    else:
        c_t = time.time()
    print(cntr)
    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cv2.destroyAllWindows()
