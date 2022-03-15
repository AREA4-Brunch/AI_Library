"""
    The script used to experiment, research and build models.
"""
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
PATH_MODELS = os.path.join(PATH_CWD, "./_models/curve_fit")


# =================================
# PROGRAM


def main():
    # Research:
    demoCurveFit()

    return


# ==========================================
# Functions:


def demoCurveFit():
    train_set_X, train_set_y = loadDatasetCurveFit()
    classes = ["red", "blue"]

    epoch_freq = 1000  # every 10 epochs metrics will be noted down
    metrics_tracker \
        =   MetricsTracker(
                [
                    "epoch_costs",
                    # "mini_batches_costs",
                    "param_update_ratio"
                ],
                { "accuracy": ClassificationAccuracyFunction() },
                epoch_freq=epoch_freq,
                do_log_live=True,
                #  metrics_lifetime="log"
            )

    # Load in a pre-trained, automatically saved, model
    # using model's idx in exploreModels as model's name
    # while saving to file, so the name is an integer for
    # models automatically saved in that loop
    # here I manually changed it from 2 to "best" to differentiate
    # it from whatever comes out of exploreModels
    # model_name = 2
    model_name = "best"
    saved_model_0, data_transform_params_0 \
        = loadExploredModel(10001, 0.0007, model_name)

    # one element in models_to_explore is tuple:
    # ( num_epochs_to_train_on, learning_rate_to_train_with,
    #   model, input_data_transformation_params,
    #   do_train_model,
    #   do_save_model_after_training )

    epochs_to_train_on = 10001
    # epochs_to_train_on = 11
    learning_rate_to_train_with = 0.0007

    models_to_explore = [
        # Best and only model, 10001 epochs on lr=0.0007:
        # ( epochs_to_train_on,
        #   learning_rate_to_train_with,
        #   getModel0(train_set_X.shape[0], learning_rate_to_train_with),
        #   None,  # data cleaner
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),

        # ( 2 * epochs_to_train_on,
        #   0.5 * learning_rate_to_train_with,
        #   getModel0(train_set_X.shape[0], 0.5 * learning_rate_to_train_with),
        #   None,  # data cleaner
        #   True,  # do_train_model
        #   False  # do_save_model_after_training
        # ),

        # do not train the loaded in model, just test it:
        (10, 0.0007, saved_model_0, data_transform_params_0, False, False),
    ]

    print("\n\nExploring..\n")

    exploreModels(train_set_X, train_set_y,
                  classes, None,
                  models_to_explore, metrics_tracker)

    print("Done exploring\n")


def exploreModels(train_set_X, train_set_y,
                  classes_names, data_cleaner,
                  models_to_explore, metrics_tracker):

    # Plot the costs during training
    models_costs = []

    # The Actual Model Exploring
    for model_idx, key in enumerate(models_to_explore):
        model_name = model_idx

        ( num_epochs, learning_rate,
         model, data_transform_params,
         do_train_model, do_save_model ) = key

        # set up the data cleaner for this model
        # data_cleaner.loadTransformationParams(data_transform_params)

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
                model_save_path = saveExploredModel(model, data_cleaner,
                                    num_epochs, learning_rate, model_name)

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

            # collect costs over each epoch:
            # for epoch_key, epoch_cost \
            #    in metrics_tracker["epoch_costs"].items():
            #     models_costs.append(epoch_cost)

            # collect costs over all mini batches:
            models_costs.append([model_name, []])
            if "mini_batches_costs" in metrics_tracker:
                for mini_batch_key, mini_batch_cost \
                in metrics_tracker["mini_batches_costs"].items():
                    models_costs[-1][1].append(mini_batch_cost)

        else:
            print("Skipping Training, using loaded in model")

        # Show the structure of the model
        print("\n\nModel:\n\n")
        print(model)

        # Check the Test Accuracy:
        test_accuracy = model.test(train_set_X, train_set_y,
                            ClassificationAccuracyFunction())

        # Plot the model's decision boundary
        plt.title("Model_{} acc={:.2f}%, alpha={}"\
                  .format(model_name, test_accuracy, learning_rate))
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])

        if type(model.layers[-1].activation_func) == type(Softmax()):
            plot_decision_boundary(lambda x: getOneHotDecoded_Y(model(x.T)).reshape(-1),
                                   train_set_X, train_set_y)
        else:
            plot_decision_boundary(lambda x: model(x.T).reshape(-1),
                                   train_set_X, train_set_y)

    # Done exlporing, plot the costs on all explored models that
    # were trained onto the same plot
    for el in models_costs:
        model_name = el[0]
        costs = el[1]
        # add the costs to the plt
        plt.plot(costs,
                 label="model: {}".format(model_name))

    plt.ylabel('cost')
    # plt.xlabel('every {} iterations'.format(metrics_tracker.epoch_freq))
    plt.xlabel('mini batch idx')  # since BGD with mini_batch_size=-1

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def saveExploredModel(model, data_cleaner,
                      num_epochs, learning_rate, model_name):
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


def getModel0(data_samples_num_of_features, learning_rate):
    # set the pseudo random num gen for the net's params' initialization:
    weights_initialization_seed = 1
    weights_psrng = np.random.RandomState(weights_initialization_seed)

    # layers' parameter initializers:
    param_init_he = HeInitialization(weights_psrng)

    # all layers but input layer:
    layers = [
        FullyConnectedLayer(data_samples_num_of_features, 15, ReLU(), param_init_he),
        FullyConnectedLayer(15, 12, ReLU(), param_init_he),
        FullyConnectedLayer(12, 7, ReLU(), param_init_he),
        FullyConnectedLayer(7, 2, Softmax(), param_init_he),
    ]

    model = NeuralNet(
        layers,
        CrossEntropyLoss(),
        MeanCostFunction(),
        Adam(OneOverT(learning_rate, 0.00142)),
    )

    return model


def loadDatasetCurveFit(seed=3):
    # from Ng's course:

    np.random.seed(seed)
    train_X, train_Y = datasets.make_moons(n_samples=300, noise=.2)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);

    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X.T, train_Y


def plot_decision_boundary(predict_func, X, y):
    # from Ng's course:

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    offset = 0.01

    # Generate a grid of points with given offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, offset), np.arange(y_min, y_max, offset))

    # Predict the function value for the whole grid
    A_L = predict_func(np.c_[xx.ravel(), yy.ravel()])
    A_L = A_L.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, A_L, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, edgecolors='k', cmap=plt.cm.Spectral)
    plt.show()


# ================================
# RUNNER PROGRAM


if __name__ == "__main__":
    main()
