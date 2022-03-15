"""
    Following Activation Functions have been implemented:
        relu
        leaky_relu
        tanh
        sigmoid
        softmax
"""


from .DifferentiableFunction import *

import numpy as np



class ReLU(DifferentiableFunction):
    def __init__(self):
        super().__init__(self.ReLU, self.ReLU_FirstDerivative)

    def __call__(self, Z):
        return self.ReLU(Z)

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def ReLU_FirstDerivative(self, Z):
        dZ = np.ones_like(Z)
        dZ[Z < 0] = 0
        return dZ


class LeakyReLU(DifferentiableFunction):
    def __init__(self):
        super().__init__(self.LeakyReLU, self.LeakyReLU_FirstDerivative)

    def __call__(self, Z):
        return self.LeakyReLU(Z)

    def LeakyReLU(self, Z):
        return np.maximum(0.001 * Z, Z)

    def LeakyReLU_FirstDerivative(self, Z):
        dZ = np.ones_like(Z)
        dZ[Z < 0] = 0.001
        return dZ


class tanh(DifferentiableFunction):
    def __init__(self):
        super().__init__(self.tanh, self.tanhFirstDerivative)

    def __call__(self, Z):
        return self.tanh(Z)

    def tanh(self, Z):
        return np.tanh(Z)

    def tanhFirstDerivative(self, Z):
        return 1 - np.tanh(Z) ** 2


# Optimized backprop tanh version
class tanhCached(DifferentiableFunction):
    """ tanh activation function storing pointer to its
        output which is reused in calculating 1st derivative.
    """
    # Attributes:
    # tanh_Z = None

    def __init__(self):
        super().__init__(self.tanh, self.tanhFirstDerivative)
        self.tanh_Z = None

    def __call__(self, Z):
        return self.tanh(Z)

    def tanh(self, Z):
        self.tanh_Z = np.tanh(Z)
        return self.tanh_Z

    def tanhFirstDerivative(self, Z=None):
        return 1 - self.tanh_Z ** 2


class Sigmoid(DifferentiableFunction):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoidFirstDerivative)

    def __call__(self, Z):
        return self.sigmoid(Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoidFirstDerivative(self, Z):
        sigmZ = self.sigmoid(Z)
        return sigmZ * (1 - sigmZ)


# Optimized backprop sigmoid version
class SigmoidCached(DifferentiableFunction):
    """ sigmoid activation function storing pointer to its
        output which is reused in calculating 1st derivative.
    """
    # Attributes:
    # sigm_Z = None

    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoidFirstDerivative)
        self.sigm_Z = None

    def __call__(self, Z):
        return self.sigmoid(Z)

    def sigmoid(self, Z):
        self.sigm_Z = 1 / (1 + np.exp(-Z))
        return self.sigm_Z

    def sigmoidFirstDerivative(self, Z=None):
        return self.sigm_Z * (1 - self.sigm_Z)


class Softmax(DifferentiableFunction):
    def __init__(self):
        super().__init__(self.softmax, self.softmaxFirstDerivative)

    def __call__(self, Z):
        return self.softmax(Z)

    def softmax(self, Z):
        z_exp = np.exp(Z)
        return np.divide(z_exp, np.sum(z_exp, axis=0, keepdims=True))

    def softmaxFirstDerivative(self, Z):
        msg1 = "Softmax's derivative has not yet been implemented"
        msg2 = "'dz_L = y_hat - y' could be used instead"
        raise DifferentiableFunctionException(msg1, msg2)


class noActivationFunc(DifferentiableFunction):
    """ Just passes input further, performs nothing on it.
        Used when skipping an activation function in layer.
    """
    def __init__(self):
        super().__init__(self.forward, self.firstDerivative)

    def __call__(self, Z):
        return self.forward(Z)

    def forward(self, Z):
        return Z

    def firstDerivative(self, Z):
        return np.ones_like(Z)
