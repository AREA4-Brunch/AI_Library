import numpy as np
import matplotlib.pyplot as plt

import os
import logging  # for debugging

import pickle  # for loading in CIFAR10
import gc


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
PATH_DATASET = os.path.join(PATH_CWD, "./_datasets/CIFAR10/")
PATH_CUSTOM_IMGS = os.path.join(PATH_CWD, "./_datasets/CIFAR10/custom_imgs_4_testing")
# path to preprocessing parameters for input imgs
PATH_DATA_CLEANERS = os.path.join(PATH_DATASET, "./data_cleaner")
# path for all models explored to be stored
PATH_MODELS = os.path.join(PATH_CWD, "./_models/LeNet5/")


# =================================
# PROGRAM


def main():
    # build full dataset (just once to store on disk)
    # buildCIFAR10(do_display_one_sample_img=True)

    # build small portion of the dataset instead
    # also just once to store on the disk
    num_train_imgs = 100
    num_test_imgs = 20
    # buildCIFAR10(num_train_imgs, num_test_imgs,
    #              do_display_one_sample_img=True)

    demoCIFAR10(num_train_imgs, num_test_imgs)

    return



# ==========================================
# Functions:


def demoCIFAR10(num_train_imgs=-1, num_test_imgs=-1):
    train_set_X, train_set_y, \
    test_set_X, test_set_y, \
    classes, data_cleaner = loadCIFAR10(num_train_imgs=num_train_imgs,
                                        num_test_imgs=num_test_imgs)
    # every 1 epochs metrics will be noted down,
    # accept for mini_batches_costs, they are updated with
    # mini_batch_freq
    epoch_freq = 1
    mini_batch_freq = 1
    metrics_tracker \
        =   MetricsTracker(
                [
                    "epoch_costs",
                    "mini_batches_costs",
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
    model_name = 0
    # saved_model_0, data_transform_params_0 \
    #     = loadExploredModel(3, 0.05, model_name)

    # one element in models_to_explore is tuple:
    # (num_epochs_to_train_on, learning_rate_to_train_with,
    # model, input_data_transformation_params,
    # do_train_model,
    # do_save_model_after_training)
    learning_rate_to_train_with = 0.005

    models_to_explore = [
        ( 3,  # num epochs to train on if do_train_model=True
          learning_rate_to_train_with,  # learning rate
          getLeNet5(learning_rate_to_train_with),
          data_cleaner.exportTransformationParams(),
          True,  # do_train_model
          True  # do_save_model_after_training
        ),

        # do not train the loaded in model, just test it:
        # (1, 0.01, saved_model_0, data_transform_params_0, False, False),
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
        gc.collect()  # clean up as much as possible

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
                        mini_batch_size=64,
                        mini_batch_seed=1,
                        metrics_to_track=metrics_tracker,
                        examples_axis=0)

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
            models_costs_epochs = []

            # collect costs over each epoch:
            if "epoch_costs" in metrics_tracker:
                for epoch_key, epoch_cost \
                in metrics_tracker["epoch_costs"].items():
                    models_costs_epochs.append(epoch_cost)

            # Add the epoch costs to the plt
            plt.plot(models_costs_epochs,
                     label="model: {}".format(model_name))
            # Done exlporing, plot the costs on all explored models that
            # were trained
            plt.ylabel('cost')
            plt.xlabel(f'every {metrics_tracker.epoch_freq} epochs')
            # plt.xlabel('mini batch idx')  # since BGD with mini_batch_size=-1
            legend = plt.legend(loc='upper center', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            plt.show()
            # free up space
            del models_costs_epochs

            # Add the mini batches costs to the plt
            models_costs_mini_batches = []
            # collect costs over all mini batches, unless 
            if "mini_batches_costs" in metrics_tracker:
                for mini_batch_key, mini_batch_cost \
                in metrics_tracker["mini_batches_costs"].items():
                    models_costs_mini_batches.append(mini_batch_cost)

            plt.plot(models_costs_mini_batches,
                     label="model: {}".format(model_name))
            # Done exlporing, plot the costs on all explored models that
            # were trained
            plt.ylabel('cost')
            plt.xlabel(f'every {metrics_tracker.mini_batch_freq} mini batches')
            # plt.xlabel('mini batch idx')  # since BGD with mini_batch_size=-1
            legend = plt.legend(loc='upper center', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            plt.show()
            # free up space
            del models_costs_mini_batches
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
            model.predictCustomImages(PATH_CUSTOM_IMGS, 32, 32, 3,
                                      data_cleaner, classes_names,
                                      models_input_shape=(-1, 32, 32, 3))
        except Exception as e:
            err_msg = f"Failed to predict custom imgs; model_idx: {model_idx}"
            logging.error(err_msg + "\n\tError:" + str(e))


# ==========================================
# Create the models to explore:


def getLeNet5(learning_rate):
    """
        Input to the model are imgs of shape: (32, 32, 3)
    """
    input_img_shape = (32, 32, 3)

    # set the pseudo random num gen for the net's params' initialization:
    weights_initialization_seed = 1
    weights_psrng = np.random.RandomState(weights_initialization_seed)

    # layers' parameter initializers:
    conv_param_init = SimpleInitializationConv2D(weights_psrng)
    fc_param_init = XavierInitialization(weights_psrng)

    # all layers but input layer:
    layers = [
        Conv2D_Layer(input_img_shape[2], 6, 5, tanhCached(), conv_param_init, stride=1),
        MaxPoolLayer(2, stride=2),
        Conv2D_Layer(6, 16, 5, tanhCached(), conv_param_init, stride=1),
        MaxPoolLayer(2, stride=2),
        Conv2D_Layer(16, 120, 5, tanhCached(), conv_param_init, stride=1),
        FlattenLayer(120),
        FullyConnectedLayer(120, 84, tanhCached(), fc_param_init),
        FullyConnectedLayer(84, 10, Softmax(), fc_param_init),
    ]

    model = NeuralNet(
        layers,
        CrossEntropyLoss(),
        MeanCostFunction(),
        # Adam(ExponentialDecayNumEpochs(learning_rate, 0.3)),
        MomentumGradientDescent(learning_rate, beta1=0.9),
        L2_Regularization(1e-4)
    )

    return model


# ==========================================
# Store the models used in exploreModels


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
    """ Given args are used to reconstruct name of the file in
        which the model was stored.
    """
    model_name = "{}_e={}_lr={}".format(model_name,
                                        num_epochs,
                                        learning_rate)
    model_save_path = os.path.join(PATH_MODELS,
                    "{}.pkl".format(model_name))

    model, data_transform_params = NeuralNet.load(model_save_path)
    return model, data_transform_params


# ==========================================
# Load in the dataset


def loadCIFAR10(num_train_imgs=-1, num_test_imgs=-1):
    """ Loads the standardized dataset formed by calling
        buildCIFAR10 once.
        If num imgs for given dataset is negative the full
        10k imgs dataset will be loaded in, otherwise will
        load the stored, trimmed down, version.
    """

    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    if num_train_imgs < 0:
        train_set_X = loadInNumpyArray(os.path.join(PATH_DATASET, "train_X.npy"))
        train_set_y = loadInNumpyArray(os.path.join(PATH_DATASET, "train_y.npy"))
    else:
        train_set_X = loadInNumpyArray(os.path.join(PATH_DATASET,
                                       f"train_X_{num_train_imgs}.npy"))
        train_set_y = loadInNumpyArray(os.path.join(PATH_DATASET,
                                       f"train_y_{num_train_imgs}.npy"))

    if num_test_imgs < 0:
        test_set_X = loadInNumpyArray(os.path.join(PATH_DATASET, "test_X.npy"))
        test_set_y = loadInNumpyArray(os.path.join(PATH_DATASET, "test_y.npy"))
    else:
        test_set_X = loadInNumpyArray(os.path.join(PATH_DATASET,
                                      f"test_X_{num_test_imgs}.npy"))
        test_set_y = loadInNumpyArray(os.path.join(PATH_DATASET,
                                      f"test_y_{num_test_imgs}.npy"))

    logging.debug("Datasets' Shapes:")
    logging.debug("train X: {}".format(train_set_X.shape))
    logging.debug("train y: {}".format(train_set_y.shape))
    logging.debug("test X: {}".format(test_set_X.shape))
    logging.debug("test y: {}".format(test_set_y.shape))
    # count number of imgs per class in train set
    unique, counts = np.unique(train_set_y, return_counts=True)
    print(unique)
    logging.debug("num of imgs per class in train set:\n{}\n"\
                  .format(dict(zip(np.array(classes)[unique.astype(int)], counts))))
    # count number of imgs per class in test set
    unique, counts = np.unique(test_set_y, return_counts=True)
    logging.debug("num of imgs per class in train set:\n{}\n"\
                  .format(dict(zip(np.array(classes)[unique.astype(int)], counts))))

    # concatenate num train examples to data_cleaner filename
    path_data_cleaner = "{}_{}.pkl".format(PATH_DATA_CLEANERS, train_set_y.shape[1])
    with open(path_data_cleaner, "rb") as in_file:
        data_cleaner = ImageDataCleaner()
        data_cleaner.loadTransformationParams(pickle.load(in_file))

    if train_set_X.shape[0] != train_set_y.shape[1]:
        raise Exception("train set's X and y have different num of examples?")
    if test_set_X.shape[0] != test_set_y.shape[1]:
        raise Exception("test set's X and y have different num of examples?")

    return train_set_X, train_set_y, test_set_X, test_set_y, \
           classes, data_cleaner


def buildCIFAR10(num_train_imgs=-1, num_test_imgs=-1,
                 do_display_one_sample_img=False):
    """ Given num imgs will be loaded in, cleaned and stored
        in `PATH_DATASET`, if -1 all imgs for train/test set
        will be loaded in.
    """
    num_batches = 5

    train_set_X = []
    train_set_y = []
    for i in range(num_batches):
        X, y = getCIFAR10_Batch(i + 1)
        train_set_X = train_set_X + X
        train_set_y = train_set_y + y

        # if enough imgs were loaded in break out
        if num_train_imgs >= 0 and len(train_set_y) >= num_train_imgs:
            break

    # shrink the train X and y if requested
    if num_train_imgs >= 0 and len(train_set_y) > num_train_imgs:
        train_set_X = train_set_X[ : num_train_imgs]
        train_set_y = train_set_y[ : num_train_imgs]

    train_set_X = np.array(train_set_X).reshape(-1, 32, 32, 3).astype(np.float64)
    train_set_y = np.array(train_set_y).reshape(1, -1).astype(np.uint8)
    logging.debug("Loaded in training set: X{}, y{}"\
                  .format(train_set_X.shape, train_set_y.shape))

    gc.collect()  # loosen things up if possible

    # load in test dataset:
    test_set_X, test_set_y = getCIFAR10_TestSet()

    # shrink the test X and y if requested
    if num_test_imgs >= 0 and test_set_y.shape[1] > num_test_imgs:
        test_set_X = test_set_X[ : num_test_imgs]
        test_set_y = test_set_y[:, : num_test_imgs]

    gc.collect()  # loosen things up if possible

    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    # select middle img to display
    sample_idx = train_set_y.shape[1] // 2
    if do_display_one_sample_img:
        print("Displaying img of category: {}"\
              .format(classes[train_set_y[0, sample_idx]]))
        plt.imshow(train_set_X[sample_idx, :, :, :]\
                   .reshape(3, 32, 32).astype(np.uint8).transpose(1, 2, 0))
        plt.show()

    # Clean Data:
    data_cleaner = ImageDataCleaner()
    # train_set_X = data_cleaner.cleanify(train_set_X, True, should_simplify=False)
    # test_set_X = data_cleaner.cleanify(test_set_X, False, should_simplify=False)
    train_set_X = data_cleaner.cleanify(train_set_X, True, should_simplify=True)
    test_set_X = data_cleaner.cleanify(test_set_X, False, should_simplify=True)

    gc.collect()  # loosen things up if possible

    # unvectorize cleaned X
    train_set_X = train_set_X.reshape(-1, 32, 32, 3)
    test_set_X = test_set_X.reshape(-1, 32, 32, 3)

    logging.debug("Datasets' Shapes:")
    logging.debug("train X: {}".format(train_set_X.shape))
    logging.debug("train y: {}".format(train_set_y.shape))
    logging.debug("test X: {}".format(test_set_X.shape))
    logging.debug("test y: {}".format(test_set_y.shape))

    if train_set_X.shape[0] != train_set_y.shape[1]:
        raise Exception("train set's X and y have different num of examples?")
    if test_set_X.shape[0] != test_set_y.shape[1]:
        raise Exception("test set's X and y have different num of examples?")

    # store standardized data as numpy arrays as backup
    try:
        # store the data cleaner
        # concatenate num train examples to data_cleaner filename
        path_data_cleaner = "{}_{}.pkl".format(PATH_DATA_CLEANERS, train_set_y.shape[1])
        with open(path_data_cleaner, "wb") as out_file:
            pickle.dump(data_cleaner.exportTransformationParams(), out_file, -1)

        storeNumpyArrays(train_set_X,
                         os.path.join(PATH_DATASET, f"train_X_{train_set_X.shape[0]}.npy"))
        storeNumpyArrays(train_set_y,
                         os.path.join(PATH_DATASET, f"train_y_{train_set_y.shape[1]}.npy"))
        storeNumpyArrays(test_set_X,
                         os.path.join(PATH_DATASET, f"test_X_{test_set_X.shape[0]}.npy"))
        storeNumpyArrays(test_set_y,
                         os.path.join(PATH_DATASET, f"test_y_{test_set_y.shape[1]}.npy"))
    except Exception as e:
        print("Failed to create backup numpy arrays:\n{}\n".format(e))

    if do_display_one_sample_img:
        print("Displaying img of category after cleaning: {}"\
              .format(classes[train_set_y[0, sample_idx]]))
        plt.imshow(train_set_X[sample_idx])
        plt.show()

    return train_set_X, train_set_y, test_set_X, test_set_y, \
           classes, data_cleaner


def getCIFAR10_Batch(batch_idx):
    batches_path = os.path.join(PATH_DATASET,
                                f"./cifar-10-batches-py/data_batch_{batch_idx}")
    print(batches_path)
    with open(batches_path, "rb") as in_file:
        data_dict = pickle.load(in_file, encoding="bytes")

    # dict_keys: [b'batch_label', b'labels', b'data', b'filenames']

    X = list(data_dict[b'data'])
    y = data_dict[b'labels']
    return X, y


def getCIFAR10_TestSet():
    test_set_path = os.path.join(PATH_DATASET, "./cifar-10-batches-py/test_batch")
    with open(test_set_path, "rb") as in_file:
        data_dict = pickle.load(in_file, encoding="bytes")

    # dict_keys: [b'batch_label', b'labels', b'data', b'filenames']

    X = data_dict[b'data'].reshape(-1, 32, 32, 3).astype(np.float64)
    y = np.array(data_dict[b'labels']).reshape(1, -1).astype(np.uint8)

    logging.debug("Loaded in test set: X{}, y{}".format(X.shape, y.shape))
    return X, y


# ================================
# RUNNER PROGRAM


if __name__ == "__main__":
    main()
