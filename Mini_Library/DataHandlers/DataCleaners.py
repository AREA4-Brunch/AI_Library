from .DataExceptions import *

import numpy as np


class DataCleaner:
    # attributes:
    # func ptrs:
    # cleanify
    # loadTransformationParams
    # exportTransformationParams

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is DataCleaner:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(self,
                 cleanifyFunc,
                 loadTransformationParamsFunc,
                 exportTransformationParamsFunc):
        # args: data, do_recalc_transformation_params[bool]
        self.cleanify = cleanifyFunc
        self.loadTransformationParams = loadTransformationParamsFunc
        self.exportTransformationParams = exportTransformationParamsFunc

    def __call__(self, data):
        return self.cleanify(data, False)


class NumericDataCleaner(DataCleaner):
    # attributes:
    # mu = None  # data means
    # sigma_squared = None

    def __init__(self):
        super().__init__(self.cleanify,
                         self.loadTransformationParams,
                         self.exportTransformationParams)
        self.mu = None
        self.sigma_squared = None

    def exportTransformationParams(self):
        """ Returns deep copy of attributes to export. """
        params = {
                    "mu": np.copy(self.mu),
                    "sigma_squared": np.copy(self.sigma_squared)
                 }
        return params

    def loadTransformationParams(self, params):
        """ Stores deep copy of given params. """
        self.mu = np.copy(params["mu"])
        self.sigma_squared = np.copy(params["sigma_squared"])

    def cleanify(self, data, do_recalc_transformation_params=True):
        """
            Cleans given data(no copy) and returns the
            data with column vectors representing data examples.

            data should be of shape:
                (num_examples, num_features)

            do_recalc_transformation_params should be False on
            test data so to reuse normalization params from
            training data when data_type is not images.
        """
        data = data.T  # convert to (num_features, num_examples)
        data = self.standardizeData(data,
                                    do_recalc_transformation_params)
        return data

    def standardizeData(self, data, do_recalc_transformation_params):
        """ Zero centers the data and normalizes it

            data should be of shape:
                (num_features, num_examples)

            do_recalc_transformation_params should be False on
            test data so to reuse normalization params from
            training data when data_type is not images.
        """

        m = data.shape[0]  # num of features

        # centralize on the x-axis by subtracting the mean:

        if do_recalc_transformation_params:
            self.mu = np.mean(data, axis=1)

        for i in range(m):
            data[i, :] -= self.mu[i]

        # even out the sizes/variance horizontally
        # after having subtracted the mean:

        if do_recalc_transformation_params:
            self.sigma_squared = np.mean(data**2, axis=1)

        for i in range(m):
            data[i, :] /= self.sigma_squared[i]

        return data

    def PCA_Whiten(self, X, new_num_features):
        """ Performs PCA and whitens the data, inplace.
            X should not have been zero centered.

            Assume input data matrix X of size [N x D]
            (N is the number of data, D is their dimensionality)
        """
        # from: https://cs231n.github.io/neural-networks-2/

        # zero-center the data (important)
        X -= np.mean(X, axis = 0)
        # get the data covariance matrix
        cov = np.dot(X.T, X) / X.shape[0]
        U, S, V = np.linalg.svd(cov)
        Xrot = np.dot(X, U) # decorrelate the data

        # Xrot_reduced becomes [N x new_num_features]:
        # Xrot_reduced = np.dot(X, U[ :, : new_num_features])
        X = np.dot(X, U[ :, : new_num_features])
        # return X  # done with PCA

        # Whiten the data
        # divide by the eigenvalues
        # (which are square roots of the singular values)
        # Xwhite = Xrot / np.sqrt(S + 1e-5)
        X = Xrot / np.sqrt(S + 1e-5)
        return X


class ImageDataCleaner(DataCleaner):
    # Attributes:
    # for accurate stanardization of data
    # numeric_data_cleaner = None

    def __init__(self):
        super().__init__(self.cleanify,
                         self.loadTransformationParams,
                         self.exportTransformationParams)
        self.numeric_data_cleaner = None

    def setNumericDataCleaner(self, numeric_data_cleaner):
        self.numeric_data_cleaner = numeric_data_cleaner

    def exportTransformationParams(self):
        """ Exports numeric data cleaner's params if it
            has been used by cleanify call with arg
            should_simplify set to False
            and was not set to None later.
        """
        if self.numeric_data_cleaner is None:
            return None

        params = {
            "numeric_data_cleaner_params":
                self.numeric_data_cleaner.exportTransformationParams()
        }
        return params

    def loadTransformationParams(self, params=None):
        if params is None:
            return
        # load in the numeric data cleaner params

        # initialize numeric data cleaner if it has not been
        if self.numeric_data_cleaner is None:
            self.numeric_data_cleaner = NumericDataCleaner()

        self.numeric_data_cleaner.loadTransformationParams(
            params["numeric_data_cleaner_params"]
        )

    def cleanify(self, data,
                 do_recalc_transformation_params=True,
                 should_simplify=True):
        """
            Cleans given data(no copy) and returns the
            data vectorized with column vectors representing
            data examples, unless output shape was given.

            data should be of shape:
                (num_examples, num_px, num_px, num_channels)

            do_recalc_transformation_params should be False on
            test data so to reuse normalization params from
            training data when data_type is not images.

            should_simplify[bool] - if true will vectorize and
                                divide by 255, else will use
                                NumericDataCleaner().standardizeData
        """

        data = ImageDataCleaner.vectorize(data)

        if should_simplify:
            data = ImageDataCleaner.standardizeData(data)
        else:
            # to standardize percisely instead of above use:
            if self.numeric_data_cleaner is None:
                self.numeric_data_cleaner = NumericDataCleaner()
            data = self.numeric_data_cleaner\
                   .standardizeData(data,
                                    do_recalc_transformation_params)
        return data

    def standardizeData(data):
        data = data / 255.
        return data

    def vectorize(data):
        """
            Reshapes from (num_examples, num_px, num_px, num_channels)
            to (num_px * num_px * num_channels, num_examples).
        """
        # For convenience, you should reshape images of
        # shape (num_px, num_px, 3) in a numpy-array of
        # shape (num_px ∗ num_px ∗ 3, 1)
        num_examples = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]
        channels = data.shape[3]  # 3 for RGB, 1 for grayscale
        data = data.reshape(
                    (num_examples, height * width * channels)
                ).T
        return data
