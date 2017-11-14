import Net as n
import NeuralNet as nn
import numpy
import cv2
import image_conversion as ic
#EXAMPLE:
net = nn.new_net([625, 20, 20, 3]) # Makes new neural network with 625 inputs, 3 outputs, and 2 hidden layers
arr = []
'''for n in range(45, 50):
    rock = []
    img_rock = cv2.imread("paper/image"+ str(n) + ".png", 0)
    height, width = img_rock.shape[:2]
    rock = [[0 for x in range(0, width)] for x in range(0, height)]
    for col in range(0, height): #Loop through images and loop through independant pixel values of images
        for row in range(0, width):
            rock[col][row] = 0 if img_rock[col][row] < 128 else 1
    temp = ic.compress_to_25x25(rock)
    rock = []
    for y in range(0, 25):
        for x in range(0, 25):
            rock.append(temp[y][x])
    arr.append(rock)'''

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
    return weights, biases

    for x in range (1, n_layers):
        layer = []
        for y in range(layer_sizes[x]):
            counter += 1
            layer.append(float(arr[counter]))
        biases.append(layer)
    
rock = []
paper = []
scissors = []
arr = []
arr_out = []
print("Rock images...")
for x in range(0, 1928):
    print(str(x))
    img_rock = cv2.imread("rock/image" + str(x) + ".png", 0)
    height, width = img_rock.shape[:2]
    rock = [[0 for x in range(0, width)] for x in range(0, height)]
    for col in range(0, height): #Loop through images and loop through independant pixel values of images
        for row in range(0, width):
            rock[col][row] = 0 if img_rock[col][row] < 128 else 1
    temp = ic.compress_to_25x25(rock)
    rock = []
    for y in range(0, 25):
        for x in range(0, 25):
            rock.append(temp[y][x])
    arr.append(rock)
    arr_out.append([1, 0, 0])
print("Paper images....")
for x in range(0, 1923):
    print(str(x))
    img_paper = cv2.imread("paper/image" + str(x) + ".png", 0)
    height, width = img_paper.shape[:2]
    paper = [[0 for x in range(0, width)] for x in range(0, height)]
    for col in range(0, height): #Loop through images and loop through independant pixel values of images
        for row in range(0, width):
            paper[col][row] = 0 if img_paper[col][row] < 128 else 1
    temp = ic.compress_to_25x25(paper)
    paper = []
    for y in range(0, 25):
        for x in range(0, 25):
            paper.append(temp[y][x])
    arr.append(paper)
    arr_out.append([0, 1, 0])
print("Scissor images...")
for x in range(0, 1839):
    print(str(x))
    img_scissors = cv2.imread("scissors/image" + str(x) + ".png", 0)
    height, width = img_scissors.shape[:2]
    scissors = [[0 for x in range(0, width)] for x in range(0, height)]
    for col in range(0, height): #Loop through images and loop through independant pixel values of images
        for row in range(0, width):
            scissors[col][row] = 0 if img_scissors[col][row] < 128 else 1
    temp = ic.compress_to_25x25(scissors)
    scissors = []
    for y in range(0, 25):
        for x in range(0, 25):
            scissors.append(temp[y][x])
    arr.append(scissors)
    arr_out.append([0, 0, 1])
print("Training in process...")
nn.train_net(net, 100, 200, arr, arr_out, 2, 0.1)
write_to_file(net, 'weights10.txt')
