from matplotlib.pyplot import axis
import numpy as np

class AccuracyFunction:
    # attributes:
    # func ptrs:
    # calc = None

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is AccuracyFunction:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(self, calcFunc):
        """ calcFunc takes args: y_correct, y_pred
            y_pred contains probabilities, not strictly 1s or 0s
        """
        self.calc = calcFunc

    def __call__(self, y_correct, y_pred):
        return self.calc(y_correct, y_pred)


class ClassificationAccuracyFunction(AccuracyFunction):
    # attributes:
    # sample is positive if its
    # predicted prob >= than this threshold
    # prob_threshold = 0.5

    def __init__(self):
        super().__init__(self.calcAccuracyClassification)
        self.prob_threshold = 0.5

    def calcAccuracyClassification(self, Y, predicted_Y):
        """
            Returns percentage of correctly predicted examples.
            Expects train/test examples to be column vectors.
            Y can be one-hot encoded or not.
            Within each example predicted_Y has probabilities
            for each class in one-hot encoded form or just
            as single vector in non one-hot case of binary
            classification.
        """
        m = Y.shape[1]  # number of examples

        # convert the predicted_Y from probabilities to
        # vector of predicted class indices
        if predicted_Y.shape[0] > 1:
            predicted_Y = np.argmax(predicted_Y, axis=0)\
                          .reshape(1, -1)
        else:
            # it was just binary classification
            predicted_Y = (predicted_Y >= self.prob_threshold)\
                          .astype(np.uint8)

        if Y.shape[0] > 1:
            # just convert to one-hot encoded
            Y = np.argmax(Y, axis=0).reshape(1, -1)

        # count on how many samples they are different
        incorrect_count = 0
        for i in range(m):
            if Y[0, i] != predicted_Y[0, i]:
                incorrect_count += 1

        avg_error = 100. / m * incorrect_count
        avg_correct = 100 - avg_error
        return avg_correct
