import numpy as np

# Returns -1 if number is negative, 1 if positive, 0 if 0
def sign (num):
    if num < 0:
        return -1
    elif num > 0:
        return 1
    return 0

# -- TanH logistic function
class TanH:
    @staticmethod
    def func (a):
        num = np.exp(a)-np.exp(-a)
        dem = np.exp(a)+np.exp(-a)
        return num/dem

    @staticmethod
    def func_deriv(a):
        return 1 - TanH.func(a)**2


# -- Class defining the sigmoid activation function
class Sigmoid:
    @staticmethod
    # Returns sigmoid function applies to a value a by the mapping SIG: a |--> 1/(1+e^-a)
    def func(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    # Returns the sigmoid prime of a value a by the mapping SIG_P: a |--> SIG(a)*(1-SIG(a))
    def func_deriv(a):
        z = Sigmoid.func(a)
        return z*(1-z)

# -- Class defining quadratic cost
class QuadraticCost:
    # Returns the cost for given vectors output and expected by C = sum(norm_sqrd(exp-out))/2(num_tests)
    @staticmethod
    def cost(outs, expected):
        sum = 0;
        for o, e in zip(outs, expected):
            sum += (e-o)*(e-o)
        return sum/(2*len(outs))

    # Returns the cost for given network with L2 regularization
    @staticmethod
    def cost_with_L2_regularization(outs, expected, weights, lmbda, training_set_size):
        sum = QuadraticCost.cost(outs, expected)
        regularization_term = 0
        for l1 in weights:
            for l2 in l1:
                for l3 in l2:
                    regularization_term += l3*l3
        regularization_term *= lmbda/(2*training_set_size)
        return sum + regularization_term

    # Returns the error vector for the output layer by d = (a-y)*sig_p(z)
    @staticmethod
    def delta(out, exp, z):
        return (out-exp)*Sigmoid.func_prime_vec(z)

# -- Class defining cross entropy
class CrossEntropy:
    # Returns the cost for given output and expected vectors by C = sum(y*ln(a) + (1-y)*ln(1-a))/num_tests
    @staticmethod
    def cost(outs, expected):
        sum = 0
        for o, e in zip(outs, expected):
            sum += e*np.log(o) + (1-e)*np.log(1-o)
        return sum/len(outs)

    # Returns the cost with L2 regularization
    @staticmethod
    def cost_with_L2_regularization(outs, expected, weights, lmbda, training_set_size):
        sum = CrossEntropy.cost(outs, expected)
        regularization_term = 0
        for l1 in weights:
            for l2 in l1:
                for l3 in l2:
                    regularization_term += l3 * l3
        regularization_term *= lmbda / (2 * training_set_size)
        return sum + regularization_term

    # Returns the error vector for the output layer by d = a-y
    @staticmethod
    def delta(out, exp, z):
        return (out-exp)
