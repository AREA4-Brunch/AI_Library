"""
    The script used to experiment, research and build models.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics

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
PATH_DATASET = os.path.join(PATH_CWD, "./_datasets/cats_non_cats")
PATH_CUSTOM_IMGS = os.path.join(PATH_CWD, "./_datasets/cats_non_cats/custom_imgs_4_testing")
PATH_MODELS = os.path.join(PATH_CWD, "./_models/cats_vs_non_cats")


# =================================
# PROGRAM


def main():
    # Research:
    demoCatsVsNonCats()

    return


# ==========================================
# Functions:


def demoCatsVsNonCats():
    train_set_X, train_set_y, \
    test_set_X, test_set_y, \
    classes, data_cleaner = loadAndCleanCatsDataset\
                            (PATH_DATASET,
                             do_display_one_sample_img=False)
    # every 1 epochs metrics will be noted down,
    # accept for mini_batches_costs, they are updated with
    # mini_batch_freq
    epoch_freq = 1
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

    # some learning rates to try out
    # learning_rates = [0.01, 0.009, 0.0075, 0.001, 0.0001]
    # best_learning_rates = [0.0075]

    # Load in a pre-trained, automatically saved, model
    # using model's idx in exploreModels as model's name
    # while saving to file, so the name is an integer for
    # models automatically saved in that loop
    # here I manually changed it from 2 to "best" to differentiate
    # it from whatever comes out of exploreModels
    # model_name = 2
    model_name = "best"
    saved_model_0, data_transform_params_0 \
        = loadExploredModel(2501, 0.0075, model_name)

    # one element in models_to_explore is tuple:
    # (num_epochs_to_train_on, learning_rate_to_train_with,
    # model, input_data_transformation_params,
    # do_train_model,
    # do_save_model_after_training)

    epochs_to_train_on = 2501
    learning_rate_to_train_with = 0.0075

    models_to_explore = [
        # ( epochs_to_train_on,
        #   0.01,
        #   getModel0(train_set_X.shape[0], 0.01),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),
        # ( epochs_to_train_on,
        #   0.009,
        #   getModel0(train_set_X.shape[0], 0.009),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),
        # ( epochs_to_train_on,
        #   0.0075,
        #   getModel0(train_set_X.shape[0], 0.0075),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),
        # ( epochs_to_train_on,
        #   0.001,
        #   getModel0(train_set_X.shape[0], 0.001),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),
        # ( epochs_to_train_on,
        #   0.0001,
        #   getModel0(train_set_X.shape[0], 0.0001),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),

        # BEST ONE:
        # add Ng's model, train it a little, store it
        # ( epochs_to_train_on,
        #   learning_rate_to_train_with,
        #   getModel0(train_set_X.shape[0], learning_rate_to_train_with),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),

        # horrible model, just for illustration
        # ( 951,
        #   0.0075,
        #   getModel1(train_set_X.shape[0], learning_rate_to_train_with),
        #   data_cleaner.exportTransformationParams(),
        #   True,  # do_train_model
        #   True  # do_save_model_after_training
        # ),

        # train loaded in model for additional 10 epochs:
        # (10, 0.0075, saved_model_0, data_transform_params_0, True, False),

        # do not train the loaded in model, just test it:
        (2501, 0.0075, saved_model_0, data_transform_params_0, False, False),
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

            # Collect the costs from training and add to plt:
            models_costs = []

            # collect costs over each epoch or mini batches
            # epochs are of higher priority
            if "epoch_costs" in metrics_tracker:
                for epoch_key, epoch_cost \
                in metrics_tracker["epoch_costs"].items():
                    models_costs.append(epoch_cost)
            elif "mini_batches_cost" in metrics_tracker:
                for mini_batch_key, mini_batch_cost \
                 in metrics_tracker["mini_batches_costs"].items():
                    models_costs.append(mini_batch_cost)

            # add the costs to the plt
            plt.plot(models_costs,
                     label="model: {}".format(model_name))
        else:
            print("Skipping Training, using loaded in model")

        # Show the structure of the model
        print("\n\nModel:\n\n")
        print(model)

        # Check the Test Accuracy:
        test_accuracy = model.test(test_set_X, test_set_y,
                            ClassificationAccuracyFunction())
        print("\n\nTest Dataset Accuracy: {}\n\n".format(test_accuracy))

        # Predict on custom imgs / sanity check:
        print('\n' + '-' * 50 + '\n\n')
        print("Custom imgs predictions:\n\n")

        try:
            model.predictCustomImages(PATH_CUSTOM_IMGS, 64, 64, 3,
                                    data_cleaner, classes_names)
        except Exception as e:
            err_msg = f"Failed to predict custom imgs; model_idx: {model_idx}"
            logging.error(err_msg + "\n\tError:" + str(e))

    # Done exlporing, plot the costs on all explored models that
    # were trained
    plt.ylabel('cost')
    plt.xlabel('every {} iterations'.format(metrics_tracker.epoch_freq))
    # plt.xlabel('mini batch idx')  # since BGD with mini_batch_size=-1

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def getModel0(data_samples_num_of_features, learning_rate):
    """ Ng's model for cats vs non-cats problem.

        To replicate Ng's results divide by num_examples on each
        backprop step in Layers.py instead of carrying 1/m all the
        way from the dcost(cost's derivative) in propagateBackward
        in NeuralNet.py

        Should then get costs:
            Cost after iteration 0: 0.771749
            Cost after iteration 100: 0.672053
            Cost after iteration 200: 0.648263
            Cost after iteration 300: 0.611507
            Cost after iteration 400: 0.567047
            Cost after iteration 500: 0.540138
            Cost after iteration 600: 0.527930
            Cost after iteration 700: 0.465477
            Cost after iteration 800: 0.369126
            Cost after iteration 900: 0.391747
            Cost after iteration 1000: 0.315187
            Cost after iteration 1100: 0.272700
            Cost after iteration 1200: 0.237419
            Cost after iteration 1300: 0.199601
            Cost after iteration 1400: 0.189263
            Cost after iteration 1500: 0.161189
            Cost after iteration 1600: 0.148214
            Cost after iteration 1700: 0.137775
            Cost after iteration 1800: 0.129740
            Cost after iteration 1900: 0.121225
            Cost after iteration 2000: 0.113821
            Cost after iteration 2100: 0.107839
            Cost after iteration 2200: 0.102855
            Cost after iteration 2300: 0.100897
            Cost after iteration 2400: 0.092878

        Test Accuracy: 0.8
    """

    # set the pseudo random num gen for the net's params' initialization:
    weights_initialization_seed = 1
    weights_psrng = np.random.RandomState(weights_initialization_seed)

    # layers' parameter initializers:
    param_init = XavierInitialization(weights_psrng)

    # all layers but input layer:
    layers = [
        FullyConnectedLayer(data_samples_num_of_features, 20, ReLU(), param_init),
        FullyConnectedLayer(20, 7, ReLU(), param_init),
        FullyConnectedLayer(7, 5, ReLU(), param_init),
        FullyConnectedLayer(5, 1, Sigmoid(), param_init),
    ]

    model = NeuralNet(
        layers,
        CrossEntropyLoss(),
        MeanCostFunction(),
        GradientDescent(learning_rate),
    )

    return model


def getModel1(data_samples_num_of_features, learning_rate):
    # set the pseudo random num gen for the net's params' initialization:
    weights_initialization_seed = 1
    weights_psrng = np.random.RandomState(weights_initialization_seed)

    # layers' parameter initializers:
    param_init_he = XavierInitialization(weights_psrng)

    # set the pseudo random num gen for dropout
    dropout_seed = 2
    dropout_psrng = np.random.RandomState(dropout_seed)

    # all layers but input layer:
    layers = [
        FullyConnectedLayer(data_samples_num_of_features, 20, ReLU(), param_init_he),
        InvertedDropoutLayer(0.5, dropout_psrng),
        FullyConnectedLayer(20, 7, ReLU(), param_init_he),
        InvertedDropoutLayer(0.6, dropout_psrng),
        FullyConnectedLayer(7, 5, ReLU(), param_init_he),
        InvertedDropoutLayer(0.5, dropout_psrng),
        FullyConnectedLayer(5, 2, Softmax(), param_init_he),
    ]

    model = NeuralNet(
        layers,
        CrossEntropyLoss(),
        MeanCostFunction(),
        # Adam(StepDecay(learning_rate, 650, 0.5)),
        GradientDescent(learning_rate),
        L2_Regularization(0.01)
    )

    return model


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


def loadAndCleanCatsDataset(path_dataset,
                            do_display_one_sample_img=False):
    # Load in the built .npy datasets (cat/non-cat)
    train_set_X_orig, train_set_y, test_set_X_orig, \
    test_set_y, classes = loadHDF5_CatsVsNonCats(path_dataset)

    # check the dimensions of the data
    m_train = train_set_X_orig.shape[0]
    m_test = test_set_X_orig.shape[0]
    height = train_set_X_orig.shape[1]
    width = train_set_X_orig.shape[2]
    channels = train_set_X_orig.shape[3]

    logging.debug("Number of training examples: m_train = {}".format(m_train))
    logging.debug("Number of testing examples: m_test = ".format(m_test))
    logging.debug("Each image is of size: ({}, {}, {})"
                  .format(height, width, channels))

    # store loaded in data as numpy arrays as backup
    # try:
    #     storeNumpyArrays(train_set_X_orig, os.path.join(path_dataset, "train_X.npy"))
    #     storeNumpyArrays(train_set_y, os.path.join(path_dataset, "train_y.npy"))
    #     storeNumpyArrays(test_set_X_orig, os.path.join(path_dataset, "test_X.npy"))
    #     storeNumpyArrays(test_set_y, os.path.join(path_dataset, "test_y.npy"))
    # except Exception as e:
    #     print("Failed to create backup numpy arrays:\n{}\n".format(e))

    if do_display_one_sample_img:
        # take a look at the random sample before noralizing and
        # cleaning the data
        sample_idx = m_train // 2
        # sample_img = cv2.cvtColor(train_set_X_orig[sample_idx], cv2.COLOR_BGR2RGB)
        sample_img = train_set_X_orig[sample_idx]
        plt.imshow(sample_img)
        plt.show()

    # Clean Data:
    data_cleaner = ImageDataCleaner()
    # modifies dimensions to be 2D and have features as rows of matrices
    # also standardizes the data
    train_set_X = data_cleaner.cleanify(train_set_X_orig, True)
    test_set_X = data_cleaner.cleanify(test_set_X_orig, False)
    # train/test_y is just 0s and 1s need to be proper 1D arrays:
    train_set_y = train_set_y.reshape((1, m_train))
    test_set_y = test_set_y.reshape((1, m_test))

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


def loadHDF5_CatsVsNonCats(path_dataset):
    # [b'non-cat' b'cat']
    # 0 for non cat, 1 for cat

    path_X_train = os.path.join(path_dataset, "train_catvnoncat.h5")
    path_y_train = os.path.join(path_dataset, "train_catvnoncat.h5")
    path_X_test = os.path.join(path_dataset, "test_catvnoncat.h5")
    path_y_test = os.path.join(path_dataset, "test_catvnoncat.h5")

    data_key_X_train='train_set_x'
    data_key_y_train='train_set_y'
    data_key_X_test='test_set_x'
    data_key_y_test='test_set_y'

    return loadImageDataSets(path_dataset, path_X_train,
                             path_y_train, path_X_test, path_y_test,
                             data_key_X_train=data_key_X_train,
                             data_key_y_train=data_key_y_train,
                             data_key_X_test=data_key_X_test,
                             data_key_y_test=data_key_y_test)


# ================================
# RUNNER PROGRAM


if __name__ == "__main__":
    main()
