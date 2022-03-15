import numpy as np
from .DataLossFunctions import *

# X = np.ones((5, 6))
# X[[1, 3], range(2)] = 0
# print((X))
# tmp = X[[1, 3], range(2)]
# print((tmp[tmp >= 0].shape))
# print(( X[[1, 3], range(2)] [ X[[1, 3], range(2)] >= 0] ))



def eval_numerical_gradient(f, x):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the single point (inside numpy array) to
        evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # increment by h
        fxh = f(x) # evalute f(x + h)
        x[ix] = old_value # restore to previous value (very important!)

        # compute the partial derivative
        grad[ix] = (fxh - fx) / h # the slope
        it.iternext() # step to next dimension

    return grad

input_psrng = np.random.RandomState(0)

X_train = input_psrng.randn(3073, 1)
X_train[3072, :] = 1
Y_train = input_psrng.random_integers(0, 10 - 1, X_train.shape[1]).reshape(1, X_train.shape[1])
L = HingeLoss()
def CIFAR10_loss_fun(W):
    return L(W, X_train, Y_train)

W = input_psrng.randn(10, 3073) * 0.001  # random weight vector
df = eval_numerical_gradient(CIFAR10_loss_fun, W) # get the gradient
mine_df = L.calcFirstDerivative(W, X_train, Y_train)
print("Their df: {}".format(df))
print("Mine df: {}".format(mine_df))

loss_original = CIFAR10_loss_fun(W) # the original loss
print( 'original loss: %f' % (loss_original, ))

# lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
    step_size = 10 ** step_size_log
    W_new = W - step_size * df # new position in the weight space
    W_mine = W - step_size * mine_df # new position in the weight space
    loss_new = CIFAR10_loss_fun(W_new)
    loss_mine = CIFAR10_loss_fun(W_mine)
    print( 'for step size %f new loss: %f' % (step_size, loss_new))
    print( 'for step size %f mine loss: %f' % (step_size, loss_mine))

# print(s:)
# original loss: 2.200718
# for step size 1.000000e-10 new loss: 2.200652
# for step size 1.000000e-09 new loss: 2.200057
# for step size 1.000000e-08 new loss: 2.194116
# for step size 1.000000e-07 new loss: 2.135493
# for step size 1.000000e-06 new loss: 1.647802
# for step size 1.000000e-05 new loss: 2.844355
# for step size 1.000000e-04 new loss: 25.558142
# for step size 1.000000e-03 new loss: 254.086573
# for step size 1.000000e-02 new loss: 2539.370888
# for step size 1.000000e-01 new loss: 25392.214036




