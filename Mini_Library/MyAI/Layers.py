from turtle import width
from .NeuralNetExceptions import *

import numpy as np
# for logging choice of seed for unprovided psrng in dropout layer:
import logging


class Layer:
    # attributes:
    # params = {}
    # forwardPropagationFunc = None
    # backwardPropagationFunc = None
    # is_training_on = True

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is Layer:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(self, forwardPropagationFunc,
                       backwardPropagationFunc):
        self.params = {}
        self.forwardPropagationFunc = forwardPropagationFunc
        self.backwardPropagationFunc = backwardPropagationFunc
        self.setIsTrainingOn()

    def propForward(self, X):
        return self.forwardPropagationFunc(X)

    def propBackward(self, dA, A_input=None, dz=None):
        # for optimization where possible
        # skip calculating dZ by providing it
        # for InvertedDropoutLayer provide only dA
        return self.backwardPropagationFunc(dA, A_input, dz)

    def setIsTrainingOn(self, is_training_on=True):
        self.is_training_on = is_training_on


class FullyConnectedLayer(Layer):
    """ Fully Connected, Non Input, Layer of the NN. """
    # parent class attributes:
    # params = {}
    # is_training_on = True

    # attributes:
    # activation_func = None
    # param_initializer = None

    def __init__(self, num_ins, num_outs,
                 activation_func, param_initializer):
        """
            num_outs = # of neurons in this layer,
            num_ins = # of neurons/features from prev layer
        """
        super().__init__(self.propagateForward, self.propagateBackward)

        self.setActivationFunc(activation_func)
        self.setParametersInitializer(param_initializer)
        self.initializeParams(num_ins, num_outs)

    def __str__(self):
        self_string = "W shape: {}\n".format(self.params['W'].shape)
        self_string += "b shape: {}\n".format(self.params['b'].shape)

        self_string += "Activation func:\n{}\n".format(type(self.activation_func).__name__)
        self_string += "Param initializer:\n{}\n".format(type(self.param_initializer).__name__)

        if "Z" in self.params:
            self_string += "Z shape: {}\n".format(self.params['Z'].shape)
            self_string += "A shape: {}\n".format(self.params['A'].shape)

        # log the values of weights:
        # self_string += "W: {}\n".format(self.params['W'])
        # self_string += "b: {}\n".format(self.params['b'])
        # log the values of Z and A:
        # if "Z" in self.params:
        #     self_string += "Z: {}\n".format(self.params['Z'])
        #     self_string += "A: {}\n".format(self.params['A'])

        return self_string

    def initializeParams(self, num_input_features, num_neurons):
        self.params = {}
        self.params["W"] = self.param_initializer(num_input_features,
                                                  num_neurons)
        self.params["b"] = np.zeros((num_neurons, 1), dtype=np.float64)

    # Magic:

    def propagateForward(self, X):
        """ X.shape = (num_features, num_data_samples) """
        self.params["Z"] = np.dot(self.params["W"], X) + self.params["b"]
        self.params["A"] = self.activation_func(self.params["Z"])
        return self.params["A"]

    def propagateBackward(self, dA, A_prev, dZ=None):
        """
            Args:
                dA -- Gradient of the cost with respect to the
                activation of current layer (computed in previous layer)
                A_prev -- activation value from forward prop of prev layer

                dz is the gradient of linear value and is optional in case
                it was easy to calculate, therefore dA is unnecessary too.
                (e.g. in case of softmax)

            Returns:
                dA_prev -- Gradient of the cost with respect to the
                activation (of the previous layer l-1), same shape as A_prev
                dW -- Gradient of the cost with respect to W (current layer l),
                same shape as W
                db -- Gradient of the cost with respect to b (current layer l),
                same shape as b
        """
        if dZ is None:
            g_prime_of_Z = self.activation_func.calcFirstDerivative(self.params["Z"])
            dZ = np.multiply(dA, g_prime_of_Z)

        dW = np.dot(dZ, A_prev.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.params["W"].T, dZ)

        if not dA_prev.shape == A_prev.shape \
        or not dW.shape == self.params["W"].shape \
        or not db.shape == self.params["b"].shape:
            raise NeuralNetException("gradient dimensions do not match")

        return dA_prev, dW, db

    # Setters:

    def setActivationFunc(self, activation_func):
        self.activation_func = activation_func

    def setParametersInitializer(self, param_initializer):
        self.param_initializer = param_initializer


class DropoutLayer(Layer):
    # parent class attributes:
    # params = {}
    # is_training_on = True

    # attributes:
    # psrng = None

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is DropoutLayer:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return super().__new__(cls)

    def __init__(self, forwardPropagationFunc,
                       backwardPropagationFunc,
                       psrng=None):
        super().__init__(forwardPropagationFunc,
                         backwardPropagationFunc)

        self.setPSRNG(psrng)

    def setPSRNG(self, psrng=None):
        if not psrng:  # set a new, random, seed initializes psrng
            seed = np.random.randint(1, 10000)
            logging.info("Dropout seed: {}\n".format(seed))
            psrng = np.random.RandomState(seed)

        self.psrng = psrng


class InvertedDropoutLayer(DropoutLayer):
    # attributes from base classes:
    # params = {}
    # psrng = None
    # is_training_on = True

    # attributes:
    # p = 0.5
    # mask = None

    # retain_rate_p = percentage of neurons to keep
    def __init__(self, retain_rate_p=0.5, psrng=None):
        super().__init__(self.propagateForward,
                         self.propagateBackward,
                         psrng)

        self.p = retain_rate_p
        self.mask = None

    def propagateForward(self, X):
        # does not apply dropout when testing
        if not self.is_training_on:
            self.params["A"] = X
            return X

        self.mask = self.psrng.rand(X.shape[0], X.shape[1]) < self.p
        # scale the output by 1/p since if you drop half the
        # neurons sum will be twice as small therefore multiply
        # it by two to preserve it
        self.params["A"] = np.multiply(X, self.mask) / self.p
        return self.params["A"]

    def propagateBackward(self, dA, dummy1=None, dummy2=None):
        # does not apply dropout when testing
        if not self.is_training_on:
            return dA

        # output is in format dA_prev, dW, db
        return np.multiply(dA, self.mask) / self.p, None, None


class Conv2D_Layer(Layer):
    """ Classical Convolutional layer. """
    # parent class attributes:
    # params = {}
    # is_training_on = True

    # attributes:
    # activation_func = None
    # param_initializer = None
    # filter_size
    # num_filters - num of channels in layers output
    # stride
    # padding
    # padding_values = (0, 0)

    def __init__(self, num_ins, num_filters, filter_size,
                 activation_func, param_initializer,
                 stride=1, padding=(0, 0), padding_values=(0, 0)):
        """
            num_ins = # of neurons/filters/channels from prev layer
            num_filters = # of neurons/filters/channels in this layer
            param_initializer - has to be Conv2D version
            padding [tuple] = (height_pad, width_pad)
            padding_values [tuple] = (height_pad_val, width_pad_val)
                                     will pad symmetrically for height
                                     and for width
        """
        super().__init__(self.propagateForward, self.propagateBackward)

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.padding_values = padding_values
        self.setActivationFunc(activation_func)
        self.setParametersInitializer(param_initializer)
        self.initializeParams(num_filters, filter_size, num_ins)

    def __str__(self):
        self_string = f"num_filters: {self.num_filters}\n"
        self_string += f"filter_size: {self.filter_size}\n"

        self_string += f"stride: {self.stride}\n"

        self_string += f"padding(height, width): {self.padding}\n"
        self_string += f"padding_values(height, width): {self.padding_values}\n"

        self_string += "W shape: {}\n".format(self.params['W'].shape)
        self_string += "b shape: {}\n".format(self.params['b'].shape)

        self_string += "Activation func:\n{}\n".format(type(self.activation_func).__name__)
        self_string += "Param initializer:\n{}\n".format(type(self.param_initializer).__name__)

        if "Z" in self.params:
            self_string += "Z shape: {}\n".format(self.params['Z'].shape)
            self_string += "A shape: {}\n".format(self.params['A'].shape)

        # log the values of weights:
        # self_string += "W: {}\n".format(self.params['W'])
        # self_string += "b: {}\n".format(self.params['b'])
        # log the values of Z and A:
        # if "Z" in self.params:
        #     self_string += "Z: {}\n".format(self.params['Z'])
        #     self_string += "A: {}\n".format(self.params['A'])

        return self_string

    def initializeParams(self, num_filters, filter_size,
                         num_input_channels):
        self.params = {}
        self.params["W"] = self.param_initializer(filter_size, filter_size,
                                                  num_input_channels,
                                                  num_filters)
        self.params["b"] = np.zeros((1, 1, 1, num_filters), dtype=np.float64)

    # Setters:

    def setActivationFunc(self, activation_func):
        self.activation_func = activation_func

    def setParametersInitializer(self, param_initializer):
        self.param_initializer = param_initializer

    # Magic:

    def propagateForward(self, X):
        """ X is of shape: (num_examples, height_prev,
                            width_prev, num_channels_prev)
        """
        (m, height_prev, width_prev, channels_prev) = X.shape
        height = 1 + (height_prev + 2 * self.padding[0] - self.filter_size) // self.stride
        width = 1 + (width_prev + 2 * self.padding[1] - self.filter_size) // self.stride

        # initialize output volume with zeros
        self.params["Z"] = np.zeros((m, height, width, self.num_filters))

        # add padding to the prev layer's output
        X_pad = self.addPadding(X)

        for i in range(m):  # loop through all mini batch examples
            for h in range(height):  # loop through vertical axis of output
                # calc from which rows to retrive to form cur window
                row_start = h * self.stride
                row_end = row_start + self.filter_size

                for w in range(width):  # loop through each column in output
                    # calc from which cols to retrive to form cur window
                    col_start = w * self.stride
                    col_end = col_start + self.filter_size

                    # loop through each filter in this layer and apply conv
                    for c in range(self.num_filters):
                        x_window = X_pad[i, row_start : row_end, col_start : col_end, :]
                        # apply conv on whole window (on all input's channels together)
                        self.params["Z"][i, h, w, c] = self.convolutionOnWindow(x_window, c)

        self.params["A"] = self.activation_func(self.params["Z"])
        return self.params["A"]

    def propagateBackward(self, dA, A_prev, dZ=None):
        """
            Args:
                dA -- Gradient of the cost with respect to the
                activation of current layer (computed in previous layer)
                A_prev -- activation value from forward prop of prev layer

                dZ is the gradient of linear value and is optional in case
                it was easy to calculate, therefore dA is unnecessary too.
                (e.g. in case of softmax)

            Returns:
                dA_prev -- Gradient of the cost with respect to the
                activation (of the previous layer l-1), same shape as A_prev
                dW -- Gradient of the cost with respect to W (current layer l),
                same shape as W
                db -- Gradient of the cost with respect to b (current layer l),
                same shape as b
        """
        if dZ is None:
            g_prime_of_Z = self.activation_func.calcFirstDerivative(self.params["Z"])
            dZ = np.multiply(dA, g_prime_of_Z)

        if dZ.shape[3] != self.num_filters:
            raise NeuralNetException("Gradient dZ has invalid channel size.")

        (m, height_prev, width_prev, channels_prev) = A_prev.shape
        (m, height, width, self.num_filters) = dZ.shape

        dW = np.zeros_like(self.params["W"])
        db = np.zeros_like(self.params["b"])
        dA_prev = np.zeros_like(A_prev)  # unpadded

        # pad the output of prev layer and its gradient
        A_prev_pad = self.addPadding(A_prev)
        dA_prev_pad = self.addPadding(dA_prev)

        for i in range(m):  # loop through all examples in mini-batch
            for h in range(height):  # loop through vertical axis of dZ
                # calc from which rows to retrive to form cur window
                row_start = h * self.stride
                row_end = row_start + self.filter_size

                for w in range(width):  # loop through each column in dZ
                    # calc from which cols to retrive to form cur window
                    col_start = w * self.stride
                    col_end = col_start + self.filter_size

                    for c in range(self.num_filters):
                        # get the window of i-th example
                        window =  A_prev_pad[i, row_start : row_end, col_start : col_end, :]

                        # add to the gradients of current filter's params and
                        # window output of prev layer
                        dW[:, :, :, c] += window * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]
                        dA_prev_pad[i, row_start : row_end, col_start : col_end, :] \
                            += self.params["W"][:, :, :, c] * dZ[i, h, w, c]

            # unpad the gradient for prev layer's output of i-th example
            dA_prev[i, :, :, :] = self.removePadding(dA_prev_pad[i])

        return dA_prev, dW, db

    def convolutionOnWindow(self, X_window, filter_idx):
        """ Returns scalar value.

            Args:
                X_window - input window of shape:
                    (filter_size, filter_size, num_channels_prev)
        """

        window = np.multiply(X_window, self.params["W"][:, :, :, filter_idx])
        # cast bias to float to get Z to be scalar
        Z = np.sum(window) + float(self.params["b"][:, :, :, filter_idx])
        return Z

    def addPadding(self, X):
        """ X should be of shape: (num_examples, height, width, channels) """
        if self.padding == (0, 0):
            return X

        # add padding only to axis corresponding to height and width
        height_padding = (self.padding[0], self.padding[0])
        width_padding = (self.padding[1], self.padding[1])
        return np.pad(X, ((0, 0), height_padding, width_padding, (0, 0)),
                      mode="constant", constant_values=self.padding_values)

    def removePadding(self, X):
        if self.padding[0] != 0 and self.padding[1] != 0:
            return X[self.padding[0] : -self.padding[0],
                     self.padding[1] : -self.padding[1], :]

        if self.padding[0] != 0:
            return X[self.padding[0] : -self.padding[0]]

        if self.padding[1] != 0:
            return X[ :, self.padding[1] : -self.padding[1], :]

        return X


class PoolingLayer(Layer):
    """ Base class for pooling layers. """
    # Parent class attributes:
    # params = {}
    # is_training_on = True

    # Attributes:
    # pool_func - function with __call__, applied on open windows
    # filter_size
    # stride

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is PoolingLayer:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return super().__new__(cls)

    def __init__(self, pool_func, propBackwardFunc, filter_size, stride=1):
        """
            pool_func - function with __call__ that will be applied
                        on each open window
        """
        super().__init__(self.propagateForward, propBackwardFunc)

        self.pool_func = pool_func
        self.filter_size = filter_size
        self.stride = stride

    def __str__(self):
        self_string = "Pooling func:\n{}\n".format(type(self.pool_func).__name__)
        self_string += f"filter_size: {self.filter_size}\n"
        self_string += f"stride: {self.stride}\n"

        if "A" in self.params:
            self_string += "A shape: {}\n".format(self.params['A'].shape)

        # log the value of A:
        # if "A" in self.params:
        #     self_string += "A: {}\n".format(self.params['A'])

        return self_string

    # Magic:

    def propagateForward(self, X):
        """ X is of shape: (num_examples, height_prev,
                            width_prev, num_channels_prev)
        """
        (m, height_prev, width_prev, channels) = X.shape
        height = 1 + (height_prev - self.filter_size) // self.stride
        width = 1 + (width_prev - self.filter_size) // self.stride

        # initialize output volume with zeros
        self.params["A"] = np.zeros((m, height, width, channels))

        for i in range(m):  # loop through all mini batch examples
            for h in range(height):  # loop through vertical axis of input
                # calc from which rows to retrive to form cur window
                row_start = h * self.stride
                row_end = row_start + self.filter_size

                for w in range(width):  # loop through each column of input
                    # calc from which cols to retrive to form cur window
                    col_start = w * self.stride
                    col_end = col_start + self.filter_size

                    # loop through each channel of input
                    for c in range(channels):
                        x_window = X[i, row_start : row_end, col_start : col_end, c]
                        self.params["A"][i, h, w, c] = self.pool_func(x_window)

        return self.params["A"]


class MaxPoolLayer(PoolingLayer):

    def __init__(self, filter_size, stride=1):
        super().__init__(np.max, self.propagateBackward, filter_size, stride)

    def propagateBackward(self, dA, A_prev, dummy=None):
        """
            Args:
                dA -- Gradient of the cost with respect to the
                activation of current layer (computed in previous layer)
                A_prev -- activation value from forward prop of prev layer

            Returns:
                dA_prev -- Gradient of the cost with respect to the
                activation (of the previous layer l-1), same shape as A_prev
                dW -- returns None since layer has no params
                db -- returns None since layer has no params
        """
        # (m, height_prev, width_prev, channels_prev) = A_prev.shape
        (m, height, width, channels) = dA.shape

        # initialize dA_prev with zeros
        dA_prev = np.zeros_like(A_prev)

        for i in range(m):  # loop through all examples in mini-batch
            for h in range(height):  # loop through vertical axis of dA
                # calc from which rows to retrive to form cur window
                row_start = h * self.stride
                row_end = row_start + self.filter_size

                for w in range(width):  # loop through each column in dA
                    # calc from which cols to retrive to form cur window
                    col_start = w * self.stride
                    col_end = col_start + self.filter_size

                    for c in range(channels):  # loop through each channel in dA
                        window = A_prev[i, row_start : row_end, col_start : col_end, c]
                        mask = MaxPoolLayer.getMask(window)
                        # add the mask to dA_prev multiplied by
                        # the rest of gradient with respect to the cost
                        dA_prev[i, row_start : row_end, col_start : col_end, c] \
                            += mask * dA[i, h, w, c]

        return dA_prev, None, None

    def getMask(window):
        """ Returns `window` shaped matrix with 1
            where max value occurs, 0 elsewhere.
        """
        return window == np.max(window)


class AvgPoolLayer(PoolingLayer):

    def __init__(self, filter_size, stride=1):
        super().__init__(np.mean, self.propagateBackward, filter_size, stride)

    def propagateBackward(self, dA, A_prev, dummy=None):
        """
            Args:
                dA -- Gradient of the cost with respect to the
                activation of current layer (computed in previous layer)
                A_prev -- activation value from forward prop of prev layer

            Returns:
                dA_prev -- Gradient of the cost with respect to the
                activation (of the previous layer l-1), same shape as A_prev
                dW -- returns None since layer has no params
                db -- returns None since layer has no params
        """
        # (m, height_prev, width_prev, channels_prev) = A_prev.shape
        (m, height, width, channels) = dA.shape

        # initialize dA_prev with zeros
        dA_prev = np.zeros_like(A_prev)

        for i in range(m):  # loop through all examples in mini-batch
            for h in range(height):  # loop through vertical axis of dA
                # calc from which rows to retrive to form cur window
                row_start = h * self.stride
                row_end = row_start + self.filter_size

                for w in range(width):  # loop through each column in dA
                    # calc from which cols to retrive to form cur window
                    col_start = w * self.stride
                    col_end = col_start + self.filter_size

                    for c in range(channels):  # loop through each channel in dA
                        dA_prev[i, row_start : row_end, col_start : col_end, c] \
                            += AvgPoolLayer.getDistributedAvg\
                                (dA[i, h, w, c], (self.filter_size, self.filter_size))

        return dA_prev, None, None

    def getDistributedAvg(pooled_value, window_shape):
        """ Distributes pooled_value(scalar) across new
            matrix of given window_shape and returns that matrix.
        """
        average = pooled_value / (window_shape[0] * window_shape[1])
        return average * np.ones(window_shape)


class FlattenLayer(Layer):
    """ Reshapes its input into shape (given_val, -1).
        Does vice versa on gradients in prop backward.
    """
    # parent class attributes:
    # params = {}
    # is_training_on = True

    # attributes:
    # num_features = None
    # old_shape = None

    def __init__(self, num_features):
        """ output will be of shape: (num_features, -1) """
        super().__init__(self.propagateForward,
                         self.propagateBackward)
        self.num_features = num_features
        self.old_shape = None

    def __str__(self):
        self_string = "Input shape: {}\n".format(self.old_shape)
        self_string += f"Output shape: ({self.num_features}, -1)\n"

        if "A" in self.params:
            self_string += "A shape: {}\n".format(self.params['A'].shape)

        # log the value of A:
        # if "A" in self.params:
        #     self_string += "A: {}\n".format(self.params['A'])

        return self_string

    def propagateForward(self, X):
        self.old_shape = X.shape
        self.params["A"] = np.reshape(X, (self.num_features, -1))
        return self.params["A"]

    def propagateBackward(self, dA, dummy1=None, dummy2=None):
        """ output is in format dA_prev, None, None """
        return np.reshape(dA, self.old_shape), None, None
