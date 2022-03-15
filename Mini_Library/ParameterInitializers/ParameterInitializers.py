import numpy as np


class ParameterInitializer:
    # attributes:
    # psrng = None

    # if psrng is not provided setd np.random as its psrng
    def __init__(self, psrng=None):
        self.setPSRNG(psrng)

    def setPSRNG(self, psrng=None):
        if not psrng:  # set a new, random, seed initializes psrng
            # seed = np.random.randint(1, 10000)
            # logging.info("Dropout seed: {}\n".format(seed))
            # psrng = np.random.RandomState(seed)
            psrng = np.random

        self.psrng = psrng


class HeInitialization(ParameterInitializer):
    # parent attributes:
    # psrng = None

    def __init__(self, psrng=None):
        super().__init__(psrng)

    def __call__(self, num_ins, num_outs):
        return self.HeInitialization(num_ins, num_outs)

    # NNs with ReLu are recommended to use He init.
    def HeInitialization(self, num_ins, num_outs):
        return self.psrng.randn(num_outs, num_ins) * np.sqrt(2.0 / num_ins)


class XavierInitialization(ParameterInitializer):
    # parent attributes:
    # psrng = None

    def __init__(self, psrng=None):
        super().__init__(psrng)

    def __call__(self, num_ins, num_outs):
        return self.XavierInitialization(num_ins, num_outs)

    def XavierInitialization(self, num_ins, num_outs):
        return self.psrng.randn(num_outs, num_ins) / np.sqrt(num_ins)


class SimpleInitialization(ParameterInitializer):
    """ Simply 0.01 * randn """
    # parent attributes:
    # psrng = None

    def __init__(self, psrng=None):
        super().__init__(psrng)

    def __call__(self, num_ins, num_outs):
        return 0.01 * self.psrng.randn(num_outs, num_ins)


class SimpleInitializationConv2D(ParameterInitializer):
    """ Simply 0.01 * randn of shape specific for Conv2D layer
        ../MyAI/Layers.py
    """
    # parent attributes:
    # psrng = None

    def __init__(self, psrng=None):
        super().__init__(psrng)

    def __call__(self, dim1, dim2, dim3, dim4):
        return 0.01 * self.psrng.randn(dim1, dim2, dim3, dim4)
        # return self.psrng.randn(dim1, dim2, dim3, dim4)
        # return self.psrng.randn(dim1, dim2, dim3, dim4) \
        #      / np.sqrt(dim1 * dim2 * dim4)
