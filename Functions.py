import math

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
    for a, b in zip(vec_a, vec_b):
        vec_c.append(coeff_1*a + coeff_2*b)
    return vec_c

# Returns the dot product between two vectors of same length, otherwise returns 0
def dot_prod (vec_a, vec_b):
    if not len(vec_a) == len(vec_b):
        return 0
    sum = 0
    for a, b in zip(vec_a, vec_b):
        sum += a*b
    return sum

# Returns the Hadamard product between two vectors
def hadamard_prod (vec_a, vec_b):
    if not len(vec_a) == len(vec_b):
        return 0
    vec = []
    for a, b in zip(vec_a, vec_b):
        vec.append(a*b)
    return vec

# Returns -1 if number is negative, 1 if positive, 0 if 0
def sign (num):
    if num < 0:
        return -1
    elif num > 0:
        return 1
    return 0

# -- Class defining the sigmoid activation function
class Sigmoid:
    @staticmethod
    # Returns sigmoid function applies to a value a by the mapping SIG: a |--> 1/(1+e^-a)
    def func(a):
        if (a > 35): return 0.99999999999999999999
        if (a < -35): return 0.00000000000000000001
        return 1 / (1 + math.exp(-a))

    @staticmethod
    # Applies the sigmoid function to all elements in a vector
    def func_vec(vec_a):
        z = []
        for x in vec_a:
            z.append(Sigmoid.func(x))
        return z

    @staticmethod
    # Returns the sigmoid prime of a value a by the mapping SIG_P: a |--> SIG(a)*(1-SIG(a))
    def func_prime(a):
        z = Sigmoid.func(a)
        return z*(1-z)

    @staticmethod
    # Applies the sigmoid prime to all elements in a vector
    def func_prime_vec(vec_a):
        sp = []
        for x in vec_a:
            sp.append(Sigmoid.func_prime(x))
        return sp

# -- Class defining the Softmax activation function
class SoftMax:
    @staticmethod
    # Applies the softmax function to a value by the mapping SOFT: a |--> a/sum(k)
    def func(val, sum):
        return val/sum

    @staticmethod
    # Applies softmax function to a vector by the mapping SOFT: a |--> e^a/sum(e^k)
    def func_vec (vec_a):
        vals = []
        sf = []
        sum = 0
        for a in vec_a:
            val = math.exp(a)
            vals.append(val)
            sum += val
        for v in vals:
            sf.append(SoftMax.func(v, sum))
        return sf

    # Returns the softmax func derivative by the mapping SOFT_P: = (a*(sum-a))/(sum^2)
    @staticmethod
    def func_prime (val, sum):
        diff = sum - val
        return (val*diff)/(sum*sum)

    # Returns the softmax prime func to all elements in a vector
    @staticmethod
    def func_prime_vec(vec_a):
        vals = SoftMax.func_vec(vec_a)
        sum = 0
        smp = []
        for x in vals:
            sum += x
        for v in vals:
            smp.append(SoftMax.func_prime(v, sum))
        return smp


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
        return hadamard_prod(add_vec(1, -1, out, exp), Sigmoid.func_prime_vec(z))

# -- Class defining cross entropy
class CrossEntropy:
    # Returns the cost for given output and expected vectors by C = sum(y*ln(a) + (1-y)*ln(1-a))/num_tests
    @staticmethod
    def cost(outs, expected):
        sum = 0
        for o, e in zip(outs, expected):
            sum += e*math.log(o) + (1-e)*math.log(1-o)
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
        return add_vec(1, -1, out, exp)
