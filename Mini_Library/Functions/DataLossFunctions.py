"""
    Following Loss Functions have been implemented:
        cross entropy (n-class)
        mean squared error (MSE)
"""


from .DifferentiableFunction import *

import numpy as np


class CrossEntropyLoss(DifferentiableFunction):
    epsilon = 1e-10  # to avoid 0 division and log(0)

    def __init__(self):
        super().__init__(self.crossEntropy,
                         self.crossEntropyFirstDerivative)

        self.epsilon = CrossEntropyLoss.epsilon

    def __call__(self, y_pred, y_correct):
        return self.crossEntropy(y_pred, y_correct)

    # can be used for softmax predictions on multi class where y_pred is
    # one hot encoded, y_correct does not have to be or can be used for
    # classical logistic regression
    # one column are class values for that columns test case
    def crossEntropy(self, y_pred, y_correct):
        if y_correct.shape[0] > 1:  # y_correct is one-hot encoded
                                    # therefore so is the y_pred
            return - np.sum(np.multiply(y_correct, np.log(y_pred)),
                            axis=0, keepdims=True)

        if y_pred.shape[0] > 1:  # it is multi class classification
            m = y_correct.shape[1]  # num of test cases
            return - np.sum(np.log(self.epsilon + y_pred[y_correct, range(m)]),
                            axis=0, keepdims=True)

        # 2-class logistic regression and nothing is one-hot encoded
        return self.LogisticRegression(y_pred, y_correct)

    # can be used for softmax predictions on multi class where y_pred is
    # one hot encoded, y_correct does not have to be or can be used for
    # classical logistic regression
    # one column are class values for that columns test case
    def crossEntropyFirstDerivative(self, y_pred, y_correct):
        m = y_correct.shape[1]

        if y_correct.shape[0] > 1:  # y_correct is one-hot encoded
            return -1. * np.sum(np.divide(y_correct, self.epsilon + y_pred),
                                axis=0, keepdims=True)

        if y_pred.shape[0] > 1:  # it is multi class classification
            dA_L = np.zeros_like(y_pred)
            dA_L[y_correct, range(m)] \
                = -1 / (self.epsilon + y_pred[y_correct, range(m)])
            return dA_L

        # 2-class logistic regression and nothing is one-hot encoded
        return self.LogisticRegressionFirstDerivative(y_pred, y_correct)

    def LogisticRegression(self, y_pred, y_correct):
        return -( np.multiply(y_correct, np.log(self.epsilon + y_pred)) \
                + np.multiply((1. - y_correct), np.log((1. + self.epsilon) - y_pred)) )

    def LogisticRegressionFirstDerivative(self, y_pred, y_correct):
        return - np.divide(y_correct, self.epsilon + y_pred) \
               + np.divide(1. - y_correct, (1. + self.epsilon) - y_pred)


class MSE_Loss(DifferentiableFunction):
    """ Mean squared error (MSE) loss function.
        Version with 0.5 multilpying loss. """

    def __init__(self):
        super().__init__(self.MSE, self.MSE_FirstDerivative)

    def __call__(self, y_pred, y_correct):
        return self.MSE(y_pred, y_correct)

    def MSE(self, y_pred, y_correct):
        return 0.5 * np.mean(np.square(y_correct - y_pred), axis=0)

    def MSE_FirstDerivative(self, y_pred, y_correct):
        return y_pred - y_correct
