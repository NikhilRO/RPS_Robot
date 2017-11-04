import Net as neuralnet

# Pass through 1D array layer_sizes, where first index is size of input and last is size of output
def new_net (layer_sizes):
    return neuralnet.Net(layer_sizes)

# Pass through neural network (net), epochs (number of training loops), training inputs (list of inputs (1D arrays))
#   expected outputs (list of expected outputs), step size, validation data (as a 2-tuple, (training inputs, expected outputs)
def train_net(net, epochs, mini_batch_size, training_inputs, expected_outputs, step_size, validation_data=None):
    net.stochastic_gradient_descent(epochs, mini_batch_size, training_inputs, expected_outputs, step_size, validation_data)

# Returns the index of the output given an input
def get_output (net, input):
    out = net.feed_forward(input)

    ind = 0
    max = out[0]
    for x in range(1, len(out)):
        if out[x] > max:
            ind = x
            max = out[x]

    return ind



#EXAMPLE:
net = new_net([200, 4, 5, 10]) # Makes new neural network with 200 inputs, 10 outputs, and 2 hidden layers

# Notes: step size around 3 is good, mini batch size at least 50-100