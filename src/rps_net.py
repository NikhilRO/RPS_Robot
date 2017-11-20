import net as n
import neural_net as nn
import numpy as np
import cv2
import image_conversion as ic
#EXAMPLE:
net = nn.new_net([625, 30, 30, 3]) # Makes new neural network with 625 inputs, 3 outputs, and 2 hidden layers
arr = []

def write_to_file (net, filename):
    file = open(filename, 'w')
    l_s = net.get_layer_sizes()
    file.write(str(len(l_s)))
    for x in l_s:
        file.write("\n" + str(x))
    w = net.get_weights()
    for a in w:
        for b in a:
            for c in b:
                file.write("\n" + str(c))

    b = net.get_biases()
    for a in b:
        for c in a:
            file.write("\n" + str(c))

    file.close()
    return "ayy"

def write_data_to_file (arr, filename):
    file = open(filename, 'a')
    file.write(str(len(arr)))
    for x in arr:
        file.write("\n" + str(x))
    file.write("\n")
    file.close()
    return "ayyy"

def read_w_b_from_file (filename):
    counter = 0
    file = open(filename, 'r')
    #read line by line
    arr = file.read().splitlines()
    n_layers = int(arr[counter])
    layer_sizes = []
    for x in range(n_layers):
        counter += 1
        layer_sizes.append(int(arr[counter]))
    biases = []
    weights = []
    for x in range(1, n_layers):
        layer = []
        for y in range(layer_sizes[x]):
            neuron = []
            for z in range(layer_sizes[x-1]):
                counter += 1
                neuron.append(float(arr[counter]))
            layer.append(neuron)
        weights.append(layer)
    for x in range (1, n_layers):
        layer = []
        for y in range(layer_sizes[x]):
            counter += 1
            layer.append(float(arr[counter]))
        biases.append(layer)
    file.close()
    return weights, biases

def read_data_from_file (filename):
    counter = 0
    data = []
    file = open(filename, 'r')
    #read line by line
    arr = file.read().splitlines()
    for y in range(len(arr)):
        if counter >= len(arr) - 1:
            counter -= 1
        length = int(float(arr[counter]))
        for x in range(length):
            counter += 1
            data.append(arr[counter])
        counter += 1
    file.close()
    return data
    
rock = []
paper = []
scissors = []
arr = []
arr_out = []

def rotate_images(move, size):
    for x in range(0, size):
        if counter % 6 == 0:
            img = cv2.imread(move + "/image" + str(x) + ".png", 0)
            M = cv2.getRotationMatrix2D((width / 2, height / 2), 12, 1)
            dst = cv2.warpAffine(img, M, (width, height))
            cv2.imwrite(move + '/image' + str(size + img_count) + '.png', dst)
            img_count += 1
        counter += 1
def init_train(move_arr, move, out_arr, size):
    print(move + " images...")
    for x in range(0, size):
        print(str(x))
        img = cv2.imread(move + "/image" + str(x) + ".png", 0)
        height, width = img.shape[:2]
        move_arr = [[0 for x in range(0, width)] for x in range(0, height)]
        for col in range(0, height): #Loop through images and loop through independant pixel values of images
            for row in range(0, width):
                move_arr[col][row] = 0 if img[col][row] < 128 else 1
        temp = ic.compress_to_25x25(move_arr)
        move_arr = []
        for y in range(0, 25):
            for x in range(0, 25):
                move_arr.append(temp[y][x])
        arr.append(move_arr)
        arr_out.append(out_arr)
'''init_train(rock, "rock", [1, 0, 0], 2919)
init_train(paper, "paper", [0, 1, 0], 2874)
init_train(scissors, "scissors", [0, 0, 1], 2755)
print("Training in process...")
nn.train_net(net, 100, 200, arr, arr_out, 1, 0.05)
write_to_file(net, 'weights13.txt')'''
