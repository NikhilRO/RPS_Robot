import random
import math

# Returns array rounded to n decimals
def arr_round (arr, decimals):
    return [round(e, decimals) for e in arr]

# Returns the addition of 2 vectors of same length, otherwise returns 0
def add_vec (coeff_1, coeff_2, vec_a, vec_b):
    if vec_a == 0 and vec_b == 0:
        return 0
    elif vec_a == 0:
        for l in range(len(vec_b)):
            vec_b[l] *= coeff_2
        return vec_b
    elif vec_b == 0:
        for l in range(len(vec_a)):
            vec_a[l] *= coeff_1
        return vec_a
    vec_c = []
    if not len(vec_a) == len(vec_b):
        return 0
    for _ in range(len(vec_b)):
        vec_c.append(coeff_1*vec_a[_] + coeff_2*vec_b[_])
    return vec_c

# Returns the dot product between two vectors of same length, otherwise returns 0
def dot_prod (vec_a, vec_b):
    if not len(vec_a) == len(vec_b):
        return 0
    sum = 0
    for _ in range(len(vec_a)):
        sum += vec_a[_]*vec_b[_]
    return sum

# Returns the Hadamard product between two vectors
def hadamard_prod (vec_a, vec_b):
    if not len(vec_a) == len(vec_b):
        return 0
    vec = []
    for _ in range(len(vec_b)):
        vec.append(vec_a[_]*vec_b[_])
    return vec

# Returns sigmoid function applies to a value a by the mapping SIG: a |--> 1/(1+e^-a)
def sigmoid(a):
    if (a > 35): return 0.99999999999999999999
    if (a < -35): return 0.00000000000000000001
    return 1 / (1 + math.exp(-a))

# Applies the sigmoid function to all elements in a vector
def sigmoid_v(vec_a):
    z = []
    for _ in range(len(vec_a)):
        z.append(sigmoid(vec_a[_]))
    return z

# Returns the sigmoid prime of a value a by the mapping SIG_P: a |--> SIG(a)*(1-SIG(a))
def sigmoid_prime(a):
    z = sigmoid(a)
    return z*(1-z)

# Applies the sigmoid prime to all elements in a vector
def sigmoid_prime_v(vec_a):
    z2 = []
    for _ in range(len(vec_a)):
        z2.append(sigmoid_prime(vec_a[_]))
    return z2

class Net:
    def __init__(self, layer_sizes):
        self.__layer_sizes = layer_sizes
        self.__n_layers = len(layer_sizes)
        # Biases is a 2D array with each layers biases. The input and output layer have no biases
        self.__biases = []
        # Weights is a 3D array w[x][y][z] where x is the layer number, y is the neuron on layer x, and z is the weight
        # connecting neuron z on layer x-1 to neuron y in layer x
        self.__weights = []
        for x in range(1, self.__n_layers):
            # self.biases.append([0 for _ in range(layer_sizes[x])])
            # self.weights.append([[0 for _ in range(layer_sizes[x - 1])] for __ in range(layer_sizes[x])])
            self.__biases.append([random.uniform(-2, 2) for _ in range(layer_sizes[x])])
            self.__weights.append([[random.uniform(-2, 2) for _ in range(layer_sizes[x-1])] for __ in range (layer_sizes[x])])

    # Returns all weights exiting from a given layer
    def get_weights(self, layer):
        return self.__weights[layer]

    # Returns all weights leading out of a neuron
    def __exiting_weights(self, layer, neuron):
        if layer >= self.__n_layers:
            return 0
        w = []
        for x in range(self.__layer_sizes[layer+1]):
            w.append(self.__weights[layer][x][neuron])
        return w

    # Returns next activation, the z value
    def __next_activation(self, curr_activations, curr_layer):
        w = self.__weights[curr_layer]

        next_layer = []
        for cw, b in zip(w, self.__biases[curr_layer]):
            next_layer.append(dot_prod(cw, curr_activations) + b)
        return next_layer

    # Returns output of neural net given an input
    def feed_forward (self, input_layer):
        for l in range(0, self.__n_layers-1):
            input_layer = sigmoid_v(self.__next_activation(input_layer, l))
        return input_layer

    #Returns fraction correct by testing it against the neural net
    def evaluate(self, test_data):
        n_tests = len(test_data)
        tests = [0 for x in range(n_tests)]
        expected = [0 for x in range(n_tests)]
        for a in range(n_tests):
            tests[a] = test_data[a][0]
            expected[a] = test_data[a][1]
        results = []
        for _ in range(n_tests):
            results.append(self.feed_forward(tests[_]))

        correct = 0.0
        for t in range(n_tests):
            max = -1
            ind = 0
            for c in range(0, len(results[t])):
                if results[t][c] > max:
                    max = results[t][c]
                    ind = c
            res = 0
            for c in range(1, len(expected[t])):
                if expected[t][c] == 1:
                    res = c
            if res == ind:
                correct += 1

        return correct/n_tests;

    # Returns the cost derivative, calculated as dCost/dActiv = activ - expect (since C = .5sum(exp-activ)^2)
    def __cost_derivative(self, output_activation, expected_val):
        return add_vec(1, -1, output_activation, expected_val)

    # Returns the next error/delta value for layer l-1 (calculated by hadamardprod(a-transpose*error, sig-prime)
    def __prev_error(self, error, exiting_weights, activation_vec_prime, layer):
        error = [dot_prod(exiting_weights[n], error) for n in range(self.__layer_sizes[layer])]
        error = hadamard_prod(error, activation_vec_prime)
        return error

    # Returns nabla_b and nabla_w, gradients of biases and weights by back propagation
    # Error (l) = hadamard(weights_transpose(l+1)*error(l+1), sig_prime(activations(l))
    def __back_prop(self, data, expected):
        z = data
        # 2D arrays storing activations of each neuron on each layer
        activation_vecs = [data]
        activation_vecs_prime = [[0 for _ in range(self.__layer_sizes[0])]]
        for l in range(0, self.__n_layers-1):
            z = self.__next_activation(z, l)
            activation_vecs.append(sigmoid_v(z))
            activation_vecs_prime.append(sigmoid_prime_v(z))
            z = sigmoid_v(z)
        # Gradient of biases is 2D and weights is 3D
        # Gradient of biases is a 2D array b[l][n] storing the bias of layer l-1 and neuron n-1
        grad_b = [0 for l in range(1, self.__n_layers)]
        # Gradient of weights is a 3D array w[l][n][p] storing the weight connecting neuron p-1 on layer l-1 to neuron n-1 on layer l
        grad_w = [[0 for n in range(self.__layer_sizes[l])] for l in range(1, self.__n_layers)]

        error = hadamard_prod(self.__cost_derivative(activation_vecs[self.__n_layers-1], expected), activation_vecs_prime[self.__n_layers-1])
        grad_b[self.__n_layers-2] = error
        for n in range(self.__layer_sizes[self.__n_layers-1]):
            grad_w[self.__n_layers-2][n] = [error[n]*activation_vecs[self.__n_layers-2][x] for x in range(self.__layer_sizes[self.__n_layers-2])]

        for l in reversed(range(1, self.__n_layers-1)):
            ex_w = []
            for n in range (self.__layer_sizes[l]):
                ex_w.append(self.__exiting_weights(l, n))
            error = self.__prev_error(error, ex_w, activation_vecs_prime[l], l)
            grad_b[l-1] = error
            for n in range(self.__layer_sizes[l]):
                grad_w[l-1][n] = [error[n] * activation_vecs[l-1][x] for x in range(self.__layer_sizes[l-1])]
        return grad_b, grad_w

    # Updates networks weights and biases based on gradients
    def __update_net_weights_biases (self, mini_batch, step_size):
        grad_b = [[0 for n in range(self.__layer_sizes[l])] for l in range(1, self.__n_layers)]
        grad_w = [[[0 for p in range(self.__layer_sizes[l-1])] for n in range(self.__layer_sizes[l])] for l in range(1, self.__n_layers)]

        for i, o in mini_batch:
            d_grad_b, d_grad_w = self.__back_prop(i, o)
            grad_b = [add_vec(1, 1, grad_b[a], d_grad_b[a]) for a in range(self.__n_layers-1)]
            grad_w = [[add_vec(1, 1, grad_w[a-1][b], d_grad_w[a-1][b]) for b in range(self.__layer_sizes[a])] for a in range(1, self.__n_layers)]
        avg_step = (step_size+0.0)/len(mini_batch)
        self.__biases = [add_vec(1, -avg_step, self.__biases[a], grad_b[a]) for a in range(self.__n_layers-1)]
        self.__weights = [[add_vec(1, -avg_step, self.__weights[a][b], grad_w[a][b]) for b in range(self.__layer_sizes[a+1])] for a in range(self.__n_layers-1)]

    # Performs SGD to network
    def stochastic_gradient_descent(self, epochs, mini_batch_size, training_inputs, expected_outputs, step_size, validation_data=None):
        tests = []
        for t in range(len(training_inputs)):
            tests.append([training_inputs[t], expected_outputs[t]])
        if len(tests) == 0:
            return

        for iters in range(epochs):
            random.shuffle(tests)
            mini_batches = [tests[curr:curr+mini_batch_size] for curr in range(0, len(tests), mini_batch_size)]
            for batch in mini_batches:
                self.__update_net_weights_biases(batch, step_size)

            if validation_data:
                print("Epoch", iters+1, " percent correct", self.evaluate(validation_data))
            else:
                print("Epoch", iters + 1, " percent correct", self.evaluate(mini_batches[0]))
            #if iters+1 == epochs:
                # print (arr_round(self.feed_forward([1,1,1]), 2)) #out 0
                # print (arr_round(self.feed_forward([0,0,0]), 2)) #out 1
                # print (arr_round(self.feed_forward([0,0,1]), 2)) #out 2
                # print (arr_round(self.feed_forward([0,1,0]), 2)) #out 3
                # print (arr_round(self.feed_forward([0,1,1]), 2)) #out 4
                # print (arr_round(self.feed_forward([1,0,0]), 2)) #out 5
                # print (arr_round(self.feed_forward([1,0,1]), 2)) #out 6
                # print (arr_round(self.feed_forward([1,1,0]), 2)) #out 7
            # print (arr_round(self.feed_forward([1, 1, 1]), 15))  # out 0
            # print (arr_round(self.feed_forward([0, 0, 0]), 15))  # out 1
            # print (arr_round(self.feed_forward([0, 0, 1]), 15))  # out 2
            # print (arr_round(self.feed_forward([0, 1, 0]), 15))  # out 3
            # print (arr_round(self.feed_forward([0, 1, 1]), 15))  # out 4
            # print (arr_round(self.feed_forward([1, 0, 0]), 15))  # out 5
            # print (arr_round(self.feed_forward([1, 0, 1]), 15))  # out 6
            # print (arr_round(self.feed_forward([1, 1, 0]), 15))  # out 7

        # for x in range(self.n_layers):
        #     print("w {0}, b {1}", self.weights[x], self.biases[x])


# testtt = [[[0,0,0],[0,1,0,0,0,0,0,0]],[[0,0,1],[0,0,1,0,0,0,0,0]],[[0,1,0],[0,0,0,1,0,0,0,0]],[[0,1,1],[0,0,0,0,1,0,0,0]],
#           [[1,0,0],[0,0,0,0,0,1,0,0]],[[1,0,1],[0,0,0,0,0,0,1,0]],[[1,1,0],[0,0,0,0,0,0,0,1]],[[1,1,1],[1,0,0,0,0,0,0,0]]]
#
# net = Net([3,20,20,8])
# step_size = 3.0
# input = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
# output = [[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0]
# ,[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0]]
#
# for a in range(10000):
#     testData = []
#     expectedResults = []
#
#     for a in range(1000):
#         r = random.randint(0,7)
#         testData.append(input[r])
#         expectedResults.append(output[r])
#
#     #print (net.feed_forward(input[0]))
#     net.stochastic_gradient_descent(100, 100, testData, expectedResults, step_size, testtt)