"""
    Following Regularization Functions have been implemented:
        L1
        L2
        Dropout
"""


from .DifferentiableFunction import *

import numpy as np


class L1_Regularization(DifferentiableFunction):
    """
        Lasso Regression.
    """
    # attributes:
    # reg_param = None  # lambda

    def __init__(self, lambda_):
        super().__init__(self.L1_Regularization, self.L1_firstDerivative)

        self.reg_param = lambda_

    def __call__(self, layers):
        return self.calc(layers)

    def L1_Regularization(self, layer_params_W):
        return 0.5 * self.reg_param * np.sum(layer_params_W)

    def L1_firstDerivative(self, layer_params_W):
        return 0.5 * self.reg_param * np.ones_like(layer_params_W)


class L2_Regularization(DifferentiableFunction):
    """
        Ridge Regression.
    """
    # attributes:
    # reg_param = None  # lambda

    def __init__(self, lambda_):
        super().__init__(self.L2_Regularization, self.L2_firstDerivative)

        self.reg_param = lambda_

    def __call__(self, layers):
        return self.calc(layers)

    def L2_Regularization(self, layer_params_W):
        return 0.5 * self.reg_param * np.sum(np.square(layer_params_W))

    def L2_firstDerivative(self, layer_params_W):
        return self.reg_param * layer_params_W
