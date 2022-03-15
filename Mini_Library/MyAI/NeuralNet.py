from .NeuralNetExceptions import *
from ..Functions.ActivationFunctions import Sigmoid, Softmax
from ..DataHandlers.ManageDatasets import loadInNumpyArray,\
                                          storeNumpyArrays, \
                                          getRandomMiniBatches, \
                                          getOneHotDecoded_Y, \
                                          getAllDirectoryImgsAsNumpyArr
import numpy as np
import pickle  # for storing the whole model
import logging


class NeuralNet:
    # attributes:
    # layers = []
    # data_loss_func = None
    # cost_func = None
    # optimizer = None
    # regularization_func = None

    def __init__(self, layers, data_loss_func,
                 cost_func, optimizer, regularization_func=None):
        """ Args:
                layers (list of Layer objects from Layers.py):
                    hidden + output layer
        """
        self.setLayers(layers)
        self.setDataLossFunction(data_loss_func)
        self.setCostFunction(cost_func)
        self.setOptimizer(optimizer)
        self.setRegularizationFunction(regularization_func)

    def __call__(self, X):
        """ Returns raw output of last layer as result of
            forward propagation.
            Propagates forward with flag is_training=False
            this flag can be used by dropout layers for example.
            Does not optimize params in the net.
        """
        y_pred = self.propagateForward(X, False)
        return y_pred

    def __str__(self):
        """ Logs names of classes of all attributes. """
        self_str = ""
        self_str += "Data loss function:\n{}\n\n"\
                    .format(type(self.data_loss_func))

        if self.regularization_func:
            self_str += "Regularization function:\n{}\n\n"\
                        .format(type(self.regularization_func))

        self_str += "Cost function:\n{}\n\n"\
                    .format(type(self.cost_func))

        self_str += "Optimizer:\n{}\n\n".format(type(self.optimizer))

        self_str += "{} Layers:\n".format(len(self.layers))

        for i, layer in enumerate(self.layers):
            # self_str += "layer #{}:\n{}\n\n".format(i, type(layer))
            self_str += "layer #{}:\n{}\n\n".format(i, layer)

        return self_str

    # Setters:

    def setLayers(self, layers):
        """ List of Layer objects in order in which they
            will appear in this neural net. Does not check
            if parameter shapes match.
        """
        self.layers = layers

    def setCostFunction(self, cost_func):
        self.cost_func = cost_func

    def setDataLossFunction(self, data_loss_func):
        self.data_loss_func = data_loss_func

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def setRegularizationFunction(self, regularization_func):
        self.regularization_func = regularization_func

    # Storing/Loading the model

    def save(self, file_path, train_data_cleaner=None):
        """ File path can be of any extension. """
        # fetch the train data transformation
        # parameters if provided
        data_transformation_params = None
        if train_data_cleaner:
            data_transformation_params \
                = train_data_cleaner.exportTransformationParams()

        with open(file_path, "wb") as out_file:
            data_to_save = [ self, data_transformation_params ]
            pickle.dump(data_to_save, out_file, -1)

    def load(file_path):
        """ Class method, returns an instance of NeuralNet
            loaded from given file_path and
            data_transformation_params if train_data_cleaner
            was provided to save otherwise None.
        """
        with open(file_path, "rb") as in_file:
            data_list = pickle.load(in_file)

        model = data_list[0]
        data_transformation_params = data_list[1]
        return model, data_transformation_params

    # Magic:

    def propagateForwAndBack(self, X, y, is_training=True):
        # propagate forward and get last  layer's activation
        A_L = self.propagateForward(X, is_training)

        # compute the loss
        losses = self.data_loss_func(A_L, y)

        # add regularization to the losses if any
        if self.regularization_func:
            regularization_loss = 0.0
            for layer in self.layers:
                # skip layers without weigths (e.g. dropout layers)
                if "W" in layer.params:
                    regularization_loss \
                        += self.regularization_func(layer.params["W"])

            regularization_loss /= losses.shape[1]
            losses += regularization_loss

        # compute the cost
        cost = self.cost_func(losses)

        # compute the gradients by propagating backward
        gradients = self.propagateBackward(A_L, X, y)

        return cost, losses, gradients

    def propagateForward(self, X, is_training):
        """ Propagates forward from X and returns activation
            of last layer.

            is_training=False will turn off the dropout.
        """
        A_prev = X
        for layer in self.layers:
            layer.setIsTrainingOn(is_training)
            A_prev = layer.propForward(A_prev)

        return A_prev

    def propagateBackward(self, losses, X, y_correct):
        """ Only computes gradients and returns dictionary of them.

            gradients are in format: "dW_L{}".format(i) where i is
                                     index of layer (0-indexed)
        """
        # derivative of loss with respect to cost func
        dcost = self.cost_func.calcFirstDerivative(losses)

        num_layers = len(self.layers)
        output_layer = self.layers[num_layers - 1]

        # use shortcut dz = y - y_hat on sigmoid and softmax
        dZ_L = None
        dloss = None
        if isinstance(output_layer.activation_func, (Softmax, Sigmoid)):
            if y_correct.shape[0] == output_layer.params["A"].shape[0]:
                # binary or multi class, where output shapes match
                dZ_L = output_layer.params["A"] - y_correct
            elif y_correct.shape[0] < output_layer.params["A"].shape[0]:
                # multi class where y is not one hot encoded
                dZ_L = np.copy(output_layer.params["A"])
                dZ_L[y_correct, range(y_correct.shape[1])] -= 1
            else:
                # y is one hot encoded, but output of last layer is not
                # => binary classification
                dZ_L = np.copy(output_layer.params["A"])
                dZ_L[np.argmax(y_correct), range(y_correct.shape[1])] -= 1

            # multiply with the 1/m from cost
            dZ_L = dcost * dZ_L
        else:
            # derivative of loss with respect to
            # last layer's activation and y_correct
            dloss = dcost * self.data_loss_func.\
                            calcFirstDerivative(output_layer.params["A"],
                                                y_correct)

        # get activation of previous layer
        if num_layers > 1:
            A_prev = self.layers[num_layers - 2].params["A"]
        else:
            A_prev = X

        dA_prev, dW, db = output_layer.\
                        propagateBackward(dloss, A_prev, dZ=dZ_L)

        # add regularization if any, only on W, not bias,
        # if the last layer does not have dropout
        if self.regularization_func and "W" in output_layer.params:
            # from https://cs231n.github.io/neural-networks-2/
            # it is not common to regularize the bias parameters
            # because they do not interact with the data through
            # multiplicative interactions, and therefore do not
            # have the interpretation of controlling the influence
            # of a data dimension on the final objective.
            dW += dcost * self.regularization_func.\
                          calcFirstDerivative(output_layer.params["W"])

        gradients = {}
        gradients["dW_L{}".format(num_layers - 1)] = dW
        gradients["db_L{}".format(num_layers - 1)] = db

        if num_layers == 1:
            return gradients

        for i in range(num_layers - 2, 0, -1):
            dA_prev, dW, db = self.layers[i].\
                propagateBackward(dA_prev,
                                  self.layers[i - 1].params["A"])
            if self.regularization_func and "W" in self.layers[i].params:
                dW += dcost * self.regularization_func.\
                        calcFirstDerivative(self.layers[i].params["W"])

            gradients["dW_L{}".format(i)] = dW
            gradients["db_L{}".format(i)] = db

        # calculate gradients of first hidden layer
        dA_prev, dW, db = self.layers[0].propagateBackward(dA_prev, X)
        if self.regularization_func and "W" in self.layers[0].params:
            dW += dcost * self.regularization_func.\
                    calcFirstDerivative(self.layers[0].params["W"])

        gradients["dW_L{}".format(0)] = dW
        gradients["db_L{}".format(0)] = db

        return gradients

    # Training/Testing:

    def test(self, X, y_correct, calc_acc_func):
        """ Runs forward propagation on X and returns:
            return calc_acc_func(y_correct, y_predictions).

            X should undergo same data preprocessing the
            training set did but with params from training set's
            preprocessing.
        """
        y_pred = self.propagateForward(X, False)
        return calc_acc_func(y_correct, y_pred)

    def train(self, X, y_correct, num_epochs,
              mini_batch_size=64, mini_batch_seed=1,
              metrics_to_track=None, examples_axis=-1):
        """ Runs the net through forward and
            backward prop and optimizes itself, repeats
            num_epochs times.

            mini_batch_size - if negative batch grad desc.
            mini_batch_seed - if negative does not shuffle

            metrics_to_track - object from MetricsTracker.py or None
                               if no metrics are to be recorded

            examples_axis - axis in X corresponding to data samples
                            default is last axis.
        """
        # negative mini_batch_size is flag for
        # batch grad desc
        if mini_batch_size < 0:
            mini_batch_size = X.shape[1]

        # initialize the optimizer object with net specific data
        self.optimizer.reinitialize(self.layers)

        for epoch in range(num_epochs):
            # logging.debug("="*20 + "\nEpoch: {}/{}".format(epoch, num_epochs - 1))
            epoch_cost = 0

            # get new shuffled set of mini batches for cur epoch            
            mini_batches = getRandomMiniBatches(X, y_correct,
                                                examples_axis,
                                                mini_batch_size,
                                                mini_batch_seed)

            for i, (cur_X, cur_y) in enumerate(mini_batches, 0):
                # just compute cost and gradients,
                # does not update weights
                cost, losses, gradients = self.propagateForwAndBack\
                                        (cur_X, cur_y)

                # updates the weights, calls onIterationEnd on
                # its learning rate object for possible learning rate
                # decays, etc.
                self.optimizer.performParameterUpdate\
                    (self.layers, gradients, metrics_to_track)

                # mini batch-wise metrics update:
                if metrics_to_track:
                    metrics_to_track.updateMetricsMiniBatch(
                        cost, losses, gradients, i, epoch,
                        y_correct, X, self, mini_batch_seed
                    )
                    # if epoch_cost is requested by metrics_to_track
                    # sum up all losses over this epoch 
                    if "epoch_costs" in metrics_to_track:
                        epoch_cost += np.sum(losses)

            # useful for specific learning rate decays, etc.
            self.optimizer.onEpochEnd()

            # epoch-wise metrics update:
            if metrics_to_track:
                if "epoch_costs" in metrics_to_track:
                    # compute the avg loss on sum of losses of all
                    # mini batches in this epoch
                    # cannot call cost_func since epoch_cost
                    # is of different shape than losses
                    # epoch_cost = self.cost_func(epoch_cost)
                    # manually calculate average
                    epoch_cost /= X.shape[examples_axis]

                metrics_to_track.updateMetricsEpoch(
                    epoch_cost, epoch, y_correct, X, self,
                    mini_batch_seed
                )

            # end of the epoch:
            # update seed so next shuffle is different
            # in case shuffling was requested (seed >= 0)
            if mini_batch_seed >= 0:
                mini_batch_seed += 1

        # clear the optimizer object
        self.optimizer.dumpCache()

        logging.debug("Done Training" + "=" * 20)

    # User friendly output/predictions:

    def predictClasses(self, X):
        """ Returns np array of shape (1, num_examples)
            of predicted class indices (0-indexed).
        """
        return getOneHotDecoded_Y(self.propagateForward(X, False))

    def predictCustomImages(self, path_arg,
                            height, width, channels=3,
                            train_data_cleaner=None,
                            classes_names=None,
                            models_input_shape=None):
        """ Returns predicted classes indices (0-indexed) on images
            stored at locations described by path_arg and their paths.
            If no imgs are successfully loaded in returns None, None.

            height and width should be dimensions of images on
            which the model was trained.

            path_arg can be a list of paths or a name of the
            directory where all the images are. If list then
            expects path_arg to be list of absolute paths to existing
            images.
            
            train_data_cleaner - instance of class from
                           ../DataHandlers/DataCleaner.py
                           should be the same one used for training

            classes_names - if provided, the predictions will be logged
                            to stdout and it should be list of strings

            models_input_shape - will try to reshape the img data to
                                 this shape, e.g: (-1, 32, 32, 3)
        """
        # get successfully loaded imgs and their paths
        test_data, test_data_imgs_paths\
            = getAllDirectoryImgsAsNumpyArr\
                (path_arg, height, width, channels)

        # note user how much data was successfully loaded
        logging.info("Custom dataset shape: {}".format(test_data.shape))

        if len(test_data_imgs_paths) == 0:
            return None, None

        # Clean it up (already of appropriate dimensions)
        if not train_data_cleaner is None:
            test_data = train_data_cleaner.cleanify(test_data, False)

        # Force a shape on test data if specified
        if not models_input_shape is None:
            test_data = test_data.reshape(models_input_shape)

        # get just the classes indices (0-indexed)
        predictions_probabilities = self.propagateForward(test_data, False)
        # predictions = self.predictClasses(test_data)
        predictions = getOneHotDecoded_Y(predictions_probabilities)

        # log predictions on collected imgs to stdout if
        # classes_names was provided
        if not classes_names is None:
            # log all predictions:
            for i in range(len(test_data_imgs_paths)):
                print("\nImage:\n{}\n".format(test_data_imgs_paths[i]))
                print("Model classifies as: {}\nwith certainty y = {}"\
                      .format(classes_names[predictions[0, i]],
                              predictions_probabilities[0, i]))

        return predictions, test_data_imgs_paths
