"""
    Following Loss Functions have been implemented:
        mean
"""

from .DifferentiableFunction import *

import numpy as np



class MeanCostFunction(DifferentiableFunction):
    def __init__(self):
        super().__init__(self.mean, self.meanFirstDerivative)

    def __call__(self, losses):
        return self.mean(losses)

    # losses is suppost to be 1D
    def mean(self, losses):
        return np.mean(losses)

    def meanFirstDerivative(self, losses):
        m = losses.shape[1]
        return 1. / m
