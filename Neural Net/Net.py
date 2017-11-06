import random
import math
import Functions

# -- The following methods are mathematical methods used by the neural network

# Returns array rounded to n decimals
def arr_round (arr, decimals):
    return [round(e, decimals) for e in arr]

# -- Class for the neural network
class Net:
    def __init__(self, layer_sizes, cost=Functions.CrossEntropy, logistic_func=Functions.Sigmoid):
        self.__layer_sizes = layer_sizes
        self.__n_layers = len(layer_sizes)
        # Biases is a 2D array with each layers biases. The input and output layer have no biases
        self.__biases = []
        # Weights is a 3D array w[x][y][z] where x is the layer number, y is the neuron on layer x, and z is the weight
        # connecting neuron z on layer x-1 to neuron y in layer x
        self.__weights = []
        for x in range(1, self.__n_layers):
            #self.__biases.append([0 for _ in range(layer_sizes[x])])
            #self.__weights.append([[0 for _ in range(layer_sizes[x - 1])] for __ in range(layer_sizes[x])])
            self.__biases.append([random.uniform(-10, 10) for _ in range(layer_sizes[x])])
            self.__weights.append([[random.uniform(-10, 10) for _ in range(layer_sizes[x-1])]
                                   for __ in range (layer_sizes[x])])
        self.__cost = cost
        self.__logistic_func = logistic_func

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
            next_layer.append(Functions.dot_prod(cw, curr_activations) + b)
        return next_layer

    # Returns output of neural net given an input
    def feed_forward (self, input_layer):
        for l in range(0, self.__n_layers-1):
            input_layer = self.__logistic_func.func_vec(self.__next_activation(input_layer, l))
        return input_layer

    # Returns the next error/delta value for layer l-1 (calculated by hadamardprod(a-transpose*error, sig-prime)
    def __prev_error(self, error, exiting_weights, activation_vec_prime, layer):
        error = [Functions.dot_prod(exiting_weights[n], error) for n in range(self.__layer_sizes[layer])]
        error = Functions.hadamard_prod(error, activation_vec_prime)
        return error

    # Returns nabla_b and nabla_w, gradients of biases and weights by back propagation
    # Error (l) = hadamard(weights_transpose(l+1)*error(l+1), sig_prime(activations(l))
    def __back_prop(self, data, expected):
        z = data
        z_s = [data]
        # 2D arrays storing activations of each neuron on each layer
        activation_vecs = [data]
        activation_vecs_prime = [[0 for _ in range(self.__layer_sizes[0])]]
        for l in range(0, self.__n_layers-1):
            z = self.__next_activation(z, l)
            z_s.append(z)
            activation_vecs.append(self.__logistic_func.func_vec(z))
            activation_vecs_prime.append(self.__logistic_func.func_prime_vec(z))
            z = self.__logistic_func.func_vec(z)
        # Gradient of biases are a 2D array and weights are a 3D array
        # Gradient of biases is a 2D array b[l][n] storing the bias of layer l-1 and neuron n-1
        grad_b = [0 for l in range(1, self.__n_layers)]
        # Gradient of weights is a 3D array w[l][n][p] storing the weight
        #   connecting neuron p-1 on layer l-1 to neuron n-1 on layer l
        grad_w = [[0 for n in range(self.__layer_sizes[l])] for l in range(1, self.__n_layers)]

        # Initial error hadamard(d_cost, sig_prime)
        error = self.__cost.delta(activation_vecs[self.__n_layers-1], expected, z_s[self.__n_layers-1])

        grad_b[self.__n_layers-2] = error
        for n in range(self.__layer_sizes[self.__n_layers-1]):
            grad_w[self.__n_layers-2][n] = [error[n]*activation_vecs[self.__n_layers-2][x]
                                            for x in range(self.__layer_sizes[self.__n_layers-2])]

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
        grad_w = [[[0 for p in range(self.__layer_sizes[l-1])]
                   for n in range(self.__layer_sizes[l])] for l in range(1, self.__n_layers)]

        for i, o in mini_batch:
            d_grad_b, d_grad_w = self.__back_prop(i, o)
            grad_b = [Functions.add_vec(1, 1, grad_b[a], d_grad_b[a]) for a in range(self.__n_layers-1)]
            grad_w = [[Functions.add_vec(1, 1, grad_w[a-1][b], d_grad_w[a-1][b])
                       for b in range(self.__layer_sizes[a])] for a in range(1, self.__n_layers)]

        avg_step = (step_size+0.0)/len(mini_batch)

        self.__biases = [Functions.add_vec(1, -avg_step, self.__biases[a], grad_b[a]) for a in range(self.__n_layers-1)]
        self.__weights = [[Functions.add_vec(1, -avg_step, self.__weights[a][b], grad_w[a][b])
                           for b in range(self.__layer_sizes[a+1])] for a in range(self.__n_layers-1)]

    # Returns fraction correct by testing it against the neural net
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

        return correct / n_tests;

    # Performs SGD to network
    def stochastic_gradient_descent(self, epochs, mini_batch_size, training_inputs, expected_outputs,
                                    step_size, test_input=None, test_output=None):
        training_data = []
        for t in range(len(training_inputs)):
            training_data.append([training_inputs[t], expected_outputs[t]])
        if len(training_data) == 0:
            return

        test_data = []
        if test_input:
            for i, o in zip(test_input, test_output):
                test_data.append([i, o])

        for iters in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[curr:curr+mini_batch_size] for curr in
                            range(0, len(training_data), mini_batch_size)]
            for batch in mini_batches:
                self.__update_net_weights_biases(batch, step_size)

            if test_input:
                print("Epoch", iters+1, " percent correct", self.evaluate(test_data))
            else:
                print("Epoch", iters+1, " percent correct", self.evaluate(mini_batches[0]))
            # if iters+1 == epochs:
            #     print (arr_round(self.feed_forward([1,1,1]), 2)) #out 0
            #     print (arr_round(self.feed_forward([0,0,0]), 2)) #out 1
            #     print (arr_round(self.feed_forward([0,0,1]), 2)) #out 2
            #     print (arr_round(self.feed_forward([0,1,0]), 2)) #out 3
            #     print (arr_round(self.feed_forward([0,1,1]), 2)) #out 4
            #     print (arr_round(self.feed_forward([1,0,0]), 2)) #out 5
            #     print (arr_round(self.feed_forward([1,0,1]), 2)) #out 6
            #     print (arr_round(self.feed_forward([1,1,0]), 2)) #out 7
            # print (arr_round(self.feed_forward([1, 1, 1]), 15))  # out 0
            # print (arr_round(self.feed_forward([0, 0, 0]), 15))  # out 1
            # print (arr_round(self.feed_forward([0, 0, 1]), 15))  # out 2
            # print (arr_round(self.feed_forward([0, 1, 0]), 15))  # out 3
            # print (arr_round(self.feed_forward([0, 1, 1]), 15))  # out 4
            # print (arr_round(self.feed_forward([1, 0, 0]), 15))  # out 5
            # print (arr_round(self.feed_forward([1, 0, 1]), 15))  # out 6
            # print (arr_round(self.feed_forward([1, 1, 0]), 15))  # out 7

    # Returns all weights in the neural network (3D Array)
    def get_weights(self):
        return self.__weights

    # Returns all biases in the neural network (2D Array)
    def get_biases(self):
        return self.__biases

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

#
# testtt = [[[0,0,0],[0,1,0,0,0,0,0,0]],[[0,0,1],[0,0,1,0,0,0,0,0]],[[0,1,0],[0,0,0,1,0,0,0,0]],[[0,1,1],[0,0,0,0,1,0,0,0]],
#           [[1,0,0],[0,0,0,0,0,1,0,0]],[[1,0,1],[0,0,0,0,0,0,1,0]],[[1,1,0],[0,0,0,0,0,0,0,1]],[[1,1,1],[1,0,0,0,0,0,0,0]]]
#
# net = Net([3,20,20,8])
# step_size = 3
# input = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
# output = [[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0]
# ,[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0]]
#
# for a in range(10000):
#     testData = []
#     expectedResults = []
#
#     for a in range(1000):
#         r = a%8
#         testData.append(input[r])
#         expectedResults.append(output[r])
#
#     #print (net.feed_forward(input[0]))
#     net.stochastic_gradient_descent(100, 100, testData, expectedResults, step_size, input, output)