"""
    Following Loss Functions have been implemented:
        cross entropy (n-class)
        mean squared error (MSE)

    To be finished:
        hinge loss for SVMs
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
            return -1. / np.sum(np.divide(y_correct, self.epsilon + y_pred),
                                axis=1, keepdims=True)

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


# EXPERIMENTAL
class HingeLoss(DifferentiableFunction):
    delta = 1

    def __init__(self):
        super().__init__(self.hingeLoss, self.hingeLossFirstDerivative)

        self.delta = HingeLoss.delta

    def __call__(self, W, X, y_correct):
        return self.hingeLoss(W, X, y_correct)

    # SVM predicts a class with the highest score/value in Wx+b
    # SVM loss, margin = delta = 1.0 since both regularization param lambda
    # and delta do the same thing - change tradeoff between
    # the data loss and the regularization loss
    # y_pred is one-hot encoded whereas y_correct is not
    # W contains bias vct and X has been added artificially 1 at end
    def hingeLoss(self, W, X, y_correct):  # y_pred = Wx + b
        self.delta = 1  # value shared with hinge loss first deriv. func.
        y_pred = np.dot(W, X)
        m = y_pred.shape[1]
        margins = np.maximum(0, y_pred - y_pred[y_correct, range(m)] + self.delta)
        margins[y_correct, range(m)] = 0
        loss = np.sum(margins, axis=0, keepdims=True)
        return loss

    # W contains bias vct and X has been added artificially 1 at end
    def hingeLossFirstDerivative(self, W, X, y_correct):
        # https://ai.stackexchange.com/questions/8281/how-do-i-calculate-the-gradient-of-the-hinge-loss-function
        # https://github.com/jayakrishna7/hinge-loss-gradient_descent/blob/master/hinge%20loss%20gradient_descent.py
        # https://stackoverflow.com/questions/40070505/gradient-descent-on-hinge-loss-svm-python-implmentation
        # https://www.youtube.com/watch?v=vi7VhPzF7YY
        # https://cs231n.github.io/optimization-1/
        # https://github.com/mark-antal-csizmadia/nn-blocks/blob/main/losses.py
        m = X.shape[1]
        # zeros of shape same as W
        # subgradient = np.zeros_like(W)

        y_pred = np.dot(W, X) # scores per class, columns are data examples
        # raw_margins = y_pred - y_pred[y_correct, range(m)] + self.delta
        # raw_margins[y_correct, range(m)] = 0

        # count the number of classes that didn’t meet the desired margin
        # (and hence contributed to the loss function) and then the
        # data vector xi scaled by this number is the gradient.
        # Notice that this is the gradient only with respect to the
        # row of W that corresponds to the correct class.
        # for i in range(X.shape[1]):  # for each example i
        #     y_pred_i = raw_margins[:, i]
        #     subgradient[y_correct[0, i]] += -X[:, i] * y_pred_i[y_pred_i > 0].shape[0]

        #     # update the rest of classes in i-th example
        #     for j in range(y_correct[0, i]):
        #         if raw_margins[j, i] > 0:
        #             subgradient[j] += X[:, i]
        #     for j in range(y_correct[0, i] + 1, W.shape[0]):
        #         if raw_margins[j, i] > 0:
        #             subgradient[j] += X[:, i]

        # return subgradient

        margins = np.maximum(0, y_pred - y_pred[y_correct, range(m)] + self.delta)
        margins[y_correct, range(m)] = 0
        margins[margins > 0] = 1
        valid_margin_count = margins.sum(axis=0)
        margins[y_correct, range(m)] -= valid_margin_count
        return margins
