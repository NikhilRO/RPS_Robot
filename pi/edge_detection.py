from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import image_conversion as ic
import neural_net as nn
import rps_net as rn
import net
import os
import gui

g = gui.GUI()

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

ROCK = 0
PAPER = 1
SCISSORS = 2

image = [[0 for x in range(width)] for y in range(height)]

#Sets the neural net's weights and biases to those obtained from training
net = nn.new_net([0])
w, b = rn.read_w_b_from_file('weights.txt')
nn.set_network_weights_biases(net, w, b)

#Changes parent directory to that containing the gifs
os.chdir("..")
os.chdir("gui images")

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

    #If a count of Rock-Paper-Scissors is played, increment the counter
    if is_play:
        if not c_p:
            cntr += 1
    c_p = is_play
    
    arr = image1.ravel()
    
    #Answer with either rock, paper or scissors on the final count
    #depending on what beats the human's move
    if cntr%4 == 0:
        answer = nn.get_output(net, arr)
        cntr=1
        if answer == ROCK:
            print("Rock", cntr)
            g.show_move('Paper.gif')
        elif answer == PAPER:
            print("Paper", cntr)
            g.show_move('Scissors.gif')
        elif answer == SCISSORS:
            print("Scissors", cntr)
            g.show_move("Rock.gif")
    elif not cntr == 1:
        if time.time() - c_t > 4:
            cntr = 1
            c_t = time.time()
        if is_play:
            g.show_move("motion.gif")
    else:
        c_t = time.time()
    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cv2.destroyAllWindows()
