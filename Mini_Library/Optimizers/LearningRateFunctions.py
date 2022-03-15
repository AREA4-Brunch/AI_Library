import numpy as np


class LearningRateFunc:
    # attributes:
    # learning_rate = None

    # ptrs to funcs:
    # reinitialize = None
    # onIterationEnd = None
    # onEpochEnd = None

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is LearningRateFunc:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(self,
                 reinitializeFunc,
                 onIterationEndFunc,
                 onEpochEndFunc,
                 learning_rate):
        # takes not parameters
        self.reinitialize = reinitializeFunc
        # takes no args:
        self.onIterationEnd = onIterationEndFunc
        # takes no args:
        self.onEpochEnd = onEpochEndFunc

        # numeric value
        self.learning_rate = learning_rate

    def __call__(self):
        return self.getLearningRate()

    def getLearningRate(self):
        return self.learning_rate


class BasicLearningRateFunc(LearningRateFunc):
    """ Just storing a constant. """
    # parent attributes:
    # learning_rate

    def __init__(self, alpha_zero):
        super().__init__(self.reinitialize,
                         self.onIterationEnd,
                         self.onEpochEnd,
                         alpha_zero)

    def reinitialize(self):
        return

    def onEpochEnd(self):
        return

    def onIterationEnd(self):
        return

    def updateLearningRate(self):
        return


class StepDecay(LearningRateFunc):
    """ Multiplies learning rate every `epoch_freq` epochs
        with `decay_rate`.
        Typical values might be reducing the learning rate
        by a half every 5 epochs, or by 0.1 every 20 epochs or
        whenever the validation error stops improving
        multiply by 0.5
    """
    # parent attributes:
    # learning_rate

    # attributes:
    # alpha_zero  # initial learning rate
    # decay_rate
    # num_epoch
    # epoch_freq

    def __init__(self, alpha_zero, epoch_freq, decay_rate):
        super().__init__(self.reinitialize,
                         self.onIterationEnd,
                         self.onEpochEnd,
                         alpha_zero)
        self.alpha_zero = alpha_zero
        self.epoch_freq = epoch_freq
        self.decay_rate = decay_rate
        self.reinitialize()

    def reinitialize(self):
        self.num_epoch = self.epoch_freq
        self.learning_rate = self.alpha_zero

    def onEpochEnd(self):
        self.num_epoch -= 1
        if self.num_epoch == 0:
            self.updateLearningRate()
            self.num_epoch = self.epoch_freq

    def onIterationEnd(self):
        return

    def updateLearningRate(self):
        """ num_epoch is zero-indexed and when equal to 0
            then learning rate is alpha zero.
        """
        self.learning_rate *= self.decay_rate


class OneOverT(LearningRateFunc):
    """ alpha = alpha_zero / (1 + decay_rate * num_iter) """
    # parent attributes:
    # learning_rate

    # attributes:
    # alpha_zero  # initial learning rate
    # decay_rate
    # num_iter

    def __init__(self, alpha_zero, decay_rate):
        super().__init__(self.reinitialize,
                         self.onIterationEnd,
                         self.onEpochEnd,
                         alpha_zero)
        self.alpha_zero = alpha_zero
        self.decay_rate = decay_rate
        self.reinitialize()

    def reinitialize(self):
        self.num_iter = 0
        self.learning_rate = self.alpha_zero

    def onEpochEnd(self):
        return

    def onIterationEnd(self):
        self.num_iter += 1
        self.updateLearningRate()

    def updateLearningRate(self):
        """ num_iter is zero-indexed and when equal to 0
            then learning rate is alpha zero.
        """
        self.learning_rate = self.alpha_zero \
                           / (1. + self.decay_rate * self.num_iter)


class ExponentialDecayNumIter(LearningRateFunc):
    """ alpha = alpha_zero * exp(-decay_rate * iter_idx) """
    # parent attributes:
    # learning_rate

    # attributes:
    # alpha_zero  # initial learning rate
    # decay_rate
    # num_iter

    def __init__(self, alpha_zero, decay_rate):
        super().__init__(self.reinitialize,
                         self.onIterationEnd,
                         self.onEpochEnd,
                         alpha_zero)
        self.alpha_zero = alpha_zero
        self.decay_rate = decay_rate
        self.reinitialize()

    def reinitialize(self):
        self.num_iter = 0
        self.learning_rate = self.alpha_zero

    def onEpochEnd(self):
        return

    def onIterationEnd(self):
        self.num_iter += 1
        self.updateLearningRate()

    def updateLearningRate(self):
        """ num_iter is zero-indexed and when equal to 0
            then learning rate is alpha zero.
        """
        self.learning_rate = self.alpha_zero \
                           * np.exp(-self.decay_rate * self.num_iter)


class ExponentialDecayNumEpochs(LearningRateFunc):
    """ alpha = alpha_zero * exp(-decay_rate * epoch_idx) """
    # parent attributes:
    # learning_rate

    # attributes:
    # alpha_zero  # initial learning rate
    # decay_rate
    # num_epoch

    def __init__(self, alpha_zero, decay_rate):
        super().__init__(self.reinitialize,
                         self.onIterationEnd,
                         self.onEpochEnd,
                         alpha_zero)
        self.alpha_zero = alpha_zero
        self.decay_rate = decay_rate
        self.reinitialize()

    def reinitialize(self):
        self.num_epoch = 0
        self.learning_rate = self.alpha_zero

    def onEpochEnd(self):
        self.num_epoch += 1
        self.updateLearningRate()

    def onIterationEnd(self):
        return

    def updateLearningRate(self):
        """ num_epoch is zero-indexed and when equal to 0
            then learning rate is alpha zero.
        """
        self.learning_rate = self.alpha_zero \
                           * np.exp(-self.decay_rate * self.num_epoch)
