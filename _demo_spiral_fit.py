import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import os
import logging  # for debugging


logging.basicConfig(level=logging.DEBUG)
# logging.getLogger(__name__).setLevel(logging.DEBUG)
# remove matplotlib font messages
logging.getLogger("matplotlib").setLevel(logging.INFO)
# logging.DEBUG
# logging.INFO
# logging.WARNING
# logging.ERROR
# logging.CRITICAL


# Import my own libraries:

from Mini_Library.MyAI.NeuralNet import *
from Mini_Library.MyAI.Layers import *
from Mini_Library.MyAI.MetricsTracker import *

from Mini_Library.Functions.AccuracyFunctions import *
from Mini_Library.Functions.ActivationFunctions import *
from Mini_Library.Functions.CostFunctions import *
from Mini_Library.Functions.DataLossFunctions import *
from Mini_Library.Functions.RegularizationFunctions import *

from Mini_Library.DataHandlers.DataCleaners import *
from Mini_Library.DataHandlers.ManageDatasets import *

from Mini_Library.Optimizers.LearningRateFunctions import *
from Mini_Library.Optimizers.Optimizers import *

from Mini_Library.ParameterInitializers.ParameterInitializers import *


# =================================
# Globals:


PATH_CWD = os.path.dirname(os.path.abspath(__file__))
# PATH_DATASETS = os.path.join(PATH_CWD, "./_datasets/")
PATH_MODELS = os.path.join(PATH_CWD, "./_models/spiral_fit")


# =================================
# PROGRAM


def main():
    demoSpiral()

    return



# ==========================================
# Functions:


def demoSpiral():
    train_set_X, train_set_y, \
    test_set_X, test_set_y, \
    classes, data_cleaner = loadSpiralDataset\
                            (do_display_one_sample_img=False)
    # every 1000 epochs metrics will be noted down,
    # accept for mini_batches_costs, they are updated with
    # mini_batch_freq
    epoch_freq = 1000
    mini_batch_freq = 1
    metrics_tracker \
        =   MetricsTracker(
                [
                    "epoch_costs",
                    # "mini_batches_costs",
                    "param_update_ratio"
                ],
                { "accuracy": ClassificationAccuracyFunction() },
                epoch_freq=epoch_freq,
                mini_batch_freq=mini_batch_freq,
                do_log_live=True,
                #  metrics_lifetime="log"
            )

    # Load in a pre-trained, automatically saved, model
    # using model's idx in exploreModels as model's name
    # while saving to file, so the name is an integer for
    # models automatically saved in that loop
    # here I manually changed it from 0 to "best" to differentiate
    # it from whatever comes out of exploreModels
    # model_name = 0
    model_name = "best"
    saved_model_0, data_transform_params_0 \
        = loadExploredModel(10001, 1e-0, model_name)

    # one element in models_to_explore is tuple:
    # (num_epochs_to_train_on, learning_rate_to_train_with,
    # model, input_data_transformation_params,
    # do_train_model,
    # do_save_model_after_training)
    learning_rate_to_train_with = 1e-0

    models_to_explore = [
        # horrible model, just for illustration
        # ( 201,  # num epochs
        #   learning_rate_to_train_with,  # learning rate
        #   getModel0(train_set_X.shape[0], learning_rate_to_train_with),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),

        # good model:
        # ( 10001,  # num epochs
        #   learning_rate_to_train_with,  # learning rate
        #   getModel1(train_set_X.shape[0], learning_rate_to_train_with),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),

        # do not train the loaded in model, just test it:
        (1, 1e-0, saved_model_0, data_transform_params_0, False, False),
    ]

    print("\n\nExploring..\n")

    exploreModels(train_set_X, train_set_y,
                  test_set_X, test_set_y,
                  classes, data_cleaner,
                  models_to_explore, metrics_tracker)

    print("\nDone exploring\n")


def exploreModels(train_set_X, train_set_y,
                  test_set_X, test_set_y,
                  classes_names, data_cleaner,
                  models_to_explore, metrics_tracker):

    # The Actual Model Exploring
    for model_idx, key in enumerate(models_to_explore):
        model_name = model_idx

        ( num_epochs, learning_rate,
          model, data_transform_params,
          do_train_model, do_save_model ) = key

        # set up the data cleaner for this model
        if not data_cleaner is None:
            data_cleaner.loadTransformationParams(data_transform_params)

        # Clear the metrics from last run:
        metrics_tracker.clear()

        # Train the model:
        if do_train_model:
            print("\n" +  "-" * 30)
            print("Training Starts\n\n")

            model.train(train_set_X, train_set_y,
                        num_epochs,
                        mini_batch_size=-1, mini_batch_seed=1,
                        metrics_to_track=metrics_tracker)

            # Save the model if requested:
            if do_save_model:
                model_save_path = saveExploredModel(model, num_epochs, learning_rate,
                                                    model_name, data_cleaner)

                # save training metrics:
                metrics_path = model_save_path[: -len(".pkl")] + "_metrics.pkl"
                with open(metrics_path, "wb") as out_file:
                    pickle.dump(metrics_tracker, out_file, -1)

            print("\n\nTraining Ended")
            print("\n" +  "-" * 30)

            # Log the parameter update ratio during training
            if "param_update_ratio" in metrics_tracker:
                print("Parameter update ratios on last mini batch:\n")
                for key, update_ratio\
                in metrics_tracker["param_update_ratio"].items():
                    print("{}: {}".format(key, update_ratio))

            # Log the training accuracy:
            if "accuracy" in metrics_tracker:
                print("\nAccuracy on training set in format:\n")
                print("e(mini_batch_epoch_shuffle_seed)_epoch_idx: accuracy\n")
                for key, accuracy in metrics_tracker["accuracy"][1].items():
                    print("{}: {}".format(key, accuracy))

            # Collect the costs from training and add to plt:
            models_costs = []

            # collect costs over each epoch:
            if "epoch_costs" in metrics_tracker:
                for epoch_key, epoch_cost \
                in metrics_tracker["epoch_costs"].items():
                    models_costs.append(epoch_cost)

            # collect costs over all mini batches:
            if "mini_batches_costs" in metrics_tracker:
                for mini_batch_key, mini_batch_cost \
                in metrics_tracker["mini_batches_costs"].items():
                    models_costs.append(mini_batch_cost)

            # add the costs to the plt
            plt.plot(models_costs,
                     label="model: {}".format(model_name))
            # Done exlporing, plot the costs on all explored models that
            # were trained
            plt.ylabel('cost')
            plt.xlabel('every {} iterations'.format(metrics_tracker.epoch_freq))
            # plt.xlabel('mini batch idx')  # since BGD with mini_batch_size=-1
            legend = plt.legend(loc='upper center', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
        else:
            print("Skipping Training, using loaded in model")

        # Show the structure of the model
        print("\n\nModel:\n\n")
        print(model)

        # Check the Test Accuracy:
        test_accuracy = model.test(test_set_X, test_set_y,
                            ClassificationAccuracyFunction())
        print("\n\nTest Dataset Accuracy: {}\n\n".format(test_accuracy))
        plot_decision_boundary(model.predictClasses, test_set_X.T, test_set_y)


def getModel0(data_samples_num_of_features, learning_rate):
    """
        Linear classifier, CS231n; example of bad net.
    """
    # set the pseudo random num gen for the net's params' initialization:
    weights_initialization_seed = 1
    weights_psrng = np.random.RandomState(weights_initialization_seed)

    # layers' parameter initializers:
    param_init = SimpleInitialization(weights_psrng)

    # all layers but input layer:
    layers = [
        FullyConnectedLayer(data_samples_num_of_features, 3, Softmax(), param_init),
    ]

    model = NeuralNet(
        layers,
        CrossEntropyLoss(),
        MeanCostFunction(),
        GradientDescent(learning_rate),
        L2_Regularization(1e-3)
    )

    return model


def getModel1(data_samples_num_of_features, learning_rate):
    """
        Non-Linear classifier, CS231n; good net.
    """
    # set the pseudo random num gen for the net's params' initialization:
    weights_initialization_seed = 1
    weights_psrng = np.random.RandomState(weights_initialization_seed)

    # layers' parameter initializers:
    param_init = SimpleInitialization(weights_psrng)

    # all layers but input layer:
    layers = [
        FullyConnectedLayer(data_samples_num_of_features, 100, ReLU(), param_init),
        FullyConnectedLayer(100, 3, Softmax(), param_init),
    ]

    model = NeuralNet(
        layers,
        CrossEntropyLoss(),
        MeanCostFunction(),
        GradientDescent(learning_rate),
        L2_Regularization(1e-3)
    )

    return model


def saveExploredModel(model, num_epochs, learning_rate,
                      model_name, data_cleaner=None):
    """ Given args are used to create name of the file in
        which the model is stored.
        Returns path to which the model was saved.
    """
    model_name = "{}_e={}_lr={}".format(model_name,
                                        num_epochs,
                                        learning_rate)
    model_save_path = os.path.join(PATH_MODELS,
                    "{}.pkl".format(model_name))

    model.save(model_save_path, data_cleaner)            
    return model_save_path


def loadExploredModel(num_epochs, learning_rate, model_name):
    """ Given args are used reconstruct name of the file in
        which the model is stored.
    """
    model_name = "{}_e={}_lr={}".format(model_name,
                                        num_epochs,
                                        learning_rate)
    model_save_path = os.path.join(PATH_MODELS,
                    "{}.pkl".format(model_name))

    model, data_transform_params = NeuralNet.load(model_save_path)
    return model, data_transform_params


def loadSpiralDataset(do_display_one_sample_img=False):
    """
        from: https://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html
    """
    psrng = np.random.RandomState(1)
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    classes = range(K)  # names of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + psrng.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    if do_display_one_sample_img:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    # Clean Data:
    data_cleaner = NumericDataCleaner()  # unused on this dataset
    m_train = X.shape[0]
    m_test = X.shape[0]
    # train_set_X = data_cleaner.cleanify(X, True)
    # test_set_X = data_cleaner.cleanify(X, False)
    train_set_X = X.T
    test_set_X = X.T
    # train/test_y is just 0s and 1s need to be proper 1D arrays:
    train_set_y = y.reshape((1, m_train))
    test_set_y = y.reshape((1, m_test))

    logging.debug("Datasets' Shapes:")
    logging.debug("train X: {}".format(train_set_X.shape))
    logging.debug("train y: {}".format(train_set_y.shape))
    logging.debug("test X: {}".format(test_set_X.shape))
    logging.debug("test y: {}".format(test_set_y.shape))

    if train_set_X.shape[1] != train_set_y.shape[1]:
        raise Exception("train set's X and y have different num of examples?")
    if test_set_X.shape[1] != test_set_y.shape[1]:
        raise Exception("test set's X and y have different num of examples?")


    return train_set_X, train_set_y, test_set_X, test_set_y, \
           classes, data_cleaner


def plot_decision_boundary(predict_func, X, y):
    """
        from: https://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html
        predict_func takes X as its only argument, returns classes
        predictions, that X is of shape (num_features, num_examples)

        X is of shape (num_examples, num_features)
    """

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    A_L = predict_func(np.c_[xx.ravel(), yy.ravel()].T)
    A_L = A_L.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, A_L, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40,
                cmap=plt.cm.Spectral, edgecolors="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #fig.savefig('spiral_linear.png')
    plt.show()
    return


# ================================
# RUNNER PROGRAM


if __name__ == "__main__":
    main()
