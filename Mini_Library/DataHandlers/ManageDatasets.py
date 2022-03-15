from .DataExceptions import *

import numpy as np
import h5py
import cv2  # pip install opencv-python
import csv  # for reading the categories names
import os  # for locating paths and files in dirs

# for displaying images in sanity checks
import matplotlib.pyplot as plt

import logging
# logging.getLogger(__file__).setLevel(logging.DEBUG)


# examples and use cases
def main():
    # initialise the .npy files
    # width = 300
    # height = 300
    # channels = 3
    # path_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                             "my_dataset")
    # Build dataset (1-time thing)
    # path_X_train, path_y_train, path_X_test, path_y_test, \
    # data_key_X_train, data_key_y_train, \
    # data_key_X_test, data_key_y_test  = buildNumpyImageDatasets(path_dataset, width,
    #                                                    height, channels,
    #                                                    dataset_type=".npy")
    # Load in the built .npy datasets
    # train_set_X_orig, train_set_y, test_set_X_orig, \
    # test_set_y, classes = loadImageDataSets(path_dataset)

    # PATH_CWD = os.path.dirname(os.path.abspath(__file__))
    # path_dataset = os.path.join(PATH_CWD, "my_dataset")

    # path_X_train = os.path.join(path_dataset, "train_catvnoncat.h5")
    # path_y_train = os.path.join(path_dataset, "train_catvnoncat.h5")
    # path_X_test = os.path.join(path_dataset, "test_catvnoncat.h5")
    # path_y_test = os.path.join(path_dataset, "test_catvnoncat.h5")

    # data_key_X_train='train_set_x'
    # data_key_y_train='train_set_y'
    # data_key_X_test='test_set_x'
    # data_key_y_test='test_set_y'

    # train_set_X_orig, train_set_y, test_set_X_orig, \
    # test_set_y, classes = loadImageDataSets(path_dataset, path_X_train,
    #         path_y_train,
    #         path_X_test, path_y_test, data_key_X_train=data_key_X_train,
    #         data_key_y_train=data_key_y_train, data_key_X_test=data_key_X_test,
    #         data_key_y_test=data_key_y_test)

    # m_train = train_set_X_orig.shape[0]
    # m_test = test_set_X_orig.shape[0]
    # # [b'non-cat' b'cat']
    # # 0 for non cat, 1 for cat
    # train_set_y = train_set_y.reshape((1, m_train))
    # test_set_y = test_set_y.reshape((1, m_test))

    # logging.debug("Shapes:")
    # logging.debug("train X: {}".format(train_set_X_orig.shape))
    # logging.debug("train y: {}".format(train_set_y.shape))
    # logging.debug("test X: {}".format(test_set_X_orig.shape))
    # logging.debug("test y: {}".format(test_set_y.shape))
    return


# Creating the datasets:


def buildNumpyImageDatasets(path_dataset, width, height, channels,
                            dataset_type=".npy"):
    """
        Creates train and test dataset out of imgs in path_dataset and
        returns paths at which the data is stored.
        Loads full X, y datasets into memory while storing.
        Returns paths to datasets created and their data keys in
        case of .h5, otherwise empty strings.

        Assumes images are stored in `/training` and `/testing` folders
        within given folder (`path_dataset`). Also assumes hierrachy described
        in saveImageDataset function.
    """
    # build training datasets
    path_train_dataset = os.path.join(path_dataset, "training")

    logging.debug("Building training dataset at:\n{}".format(path_train_dataset))

    path_X_train, path_y_train, \
    data_key_X_train, data_key_y_train = saveImageDataset(path_train_dataset, path_dataset,
                                                          height, width, channels=channels,
                                                          path_prefix="train_",
                                                          output_extension=dataset_type)

    # build testing datasets
    path_test_dataset = os.path.join(path_dataset, "testing")

    logging.debug("Done building training dataset.")
    logging.debug("Building testing dataset at:\n{}".format(path_test_dataset))

    path_X_test, path_y_test, \
    data_key_X_test, data_key_y_test = saveImageDataset(path_test_dataset, path_dataset,
                                                        height, width, channels=channels,
                                                        path_prefix="test_",
                                                        output_extension=dataset_type)

    logging.debug("Done building testing dataset.")
    logging.debug("Built numpy datasets at following locations:\n{}\n{}\n{}\n{}"
                  .format(path_X_train, path_y_train, path_X_test, path_y_test))

    return path_X_train, path_y_train, path_X_test, path_y_test, \
    data_key_X_train, data_key_y_train, data_key_X_test, data_key_y_test


def saveImageDataset(path_src_dataset, path_store_np_data,
                     width, height, channels=3,
                     path_prefix="train_",
                     output_extension=".npy"):
    """
        Returns paths where X and y data were stored as well as keys
        for .h5 files, keys are empty strings if format was .npy.
        Images are stored in BGR format.
    """

    logging.debug("Building numpy arrays from dataset:\n{}"
                  .format(path_src_dataset))
    X, y = getImageDataset(path_src_dataset, width, height,
                           channels=channels)
    # construct file output names
    if output_extension == ".npy":
        output_name_X = "{}X.npy".format(path_prefix)
        output_name_y = "{}y.npy".format(path_prefix)
        data_key_X = ''
        data_key_y = ''
    else:
        output_name_X = "{}_dataset.h5".format(path_prefix)
        output_name_y = output_name_X
        data_key_X = "{}X".format(path_prefix)
        data_key_y = "{}X".format(path_prefix)

    # store the dataset:
    output_X_path = os.path.join(path_store_np_data, output_name_X)
    output_y_path = os.path.join(path_store_np_data, output_name_y)

    logging.debug("Storing built numpy arrays:\n{}\n{}"
                  .format(output_X_path, output_y_path))
    storeNumpyArrays(X, output_X_path, data_key=data_key_X)
    storeNumpyArrays(y, output_y_path, data_key=data_key_y)
    logging.debug("Successfully stored the numpy arrays")

    return output_X_path, output_y_path, data_key_X, data_key_y


def getImageDataset(path_dataset, width, height, channels=3):
    """
        Returns X, y numpy arrays from the given
        path_dataset directory.

        While constructing arrays from images from given directory,
        images are reshaped to given height, width and channel.
        There is no shuffling.
        Shapes:
        X.shape = (num_examples, height, width, channels)
        y.shape = (1, num_examples)
        Expects following hierarchy within given dataset folder:
        GIVEN_FOLDER:
            - categories.csv containing a list with names of
                             categories separated  by comma without
                             whitespace. e.g: `Non_Cat,Cat`
                             (those same names are used for subfolders)
            - categories[0]
                only img files of only 0-th category
            - categories[1]
                only img files of only 1-th category
            ...
    """
    # categories = ["Non_Cat", "Cat"]
    categories = loadClasses(os.path.dirname(path_dataset))
    logging.debug("Found categories:\n{}".format(categories))

    # get the number of training examples in total in order
    # to allocate enough space for it
    num_examples = 0
    for category in categories:
        path_category = os.path.join(path_dataset, category)
        for path, subdirs, file_names in os.walk(path_category):
            num_examples += len(file_names)

    logging.debug("\nAttempting to save {} images..".format(num_examples))
    logging.debug("Allocating space..")
    # VERY IMPORTANT TO SET DTYPE TO UINT8 if
    # you want to display using plt.imshow
    X = np.empty([num_examples, height, width, channels],
                 dtype=np.uint8)
    # convert 0s and 1s to ints from floats:
    y = np.empty([1, num_examples], dtype=np.int8)
    # keep track of which imgs might have failed to be loaded
    # in order to skip them:
    is_used = np.ones(X.shape[0], dtype=bool)

    logging.debug("Successfully allocated space.")
    # keep track of which row in allocate matrix to overwrite
    img_idx = 0
    for cat_idx in range(len(categories)):
        category = categories[cat_idx]
        path_category = os.path.join(path_dataset, category)
        logging.debug("Processing {} imgs found in:\n{}"
                      .format(category, path_category))
        for path, subdirs, file_names in os.walk(path_category):
            for img_name in file_names:
                img_path = os.path.join(path, img_name)
                try:
                    # in case getImageAsData fails None will
                    # be returned and X[img_idx]=None will raise
                    X[img_idx] = getImageAsData(img_path, height,
                                                width, channels)
                    y[0, img_idx] = cat_idx
                except Exception as e:
                    is_used[0, img_idx] = False
                    logging.error("Failed to store an img: {}".format(img_path))
                    logging.error("Error:\n\t{}".format(e))

            img_idx += 1
            if img_idx % 1000:
                logging.debug("Processed {}th image".format(img_idx))

    # Perhaps shuffle the data after loading it in from a file and using it
    # outside this module
    # shuffle the data:
    # logging.debug("Shuffling the data..")
    # seed = 3  # so the shuffling would be the same for x, y
    # np.random.RandomState(seed).shuffle(X)
    # np.random.RandomState(seed).shuffle(y)
    # np.random.RandomState(seed).shuffle(is_used)
    # logging.debug("Done shuffling the data.")

    # sanity check:
    logging.debug("Num of imgs successfully stored: {}".
                  format(np.sum(is_used)))
    logging.debug("Y corresponding to img shown: {} ({})".
                  format(y[0, 0], categories[y[0, 0]]))
    # X.resize(1, height, width, channels)
    plt.imshow(X[0])
    plt.show()

    return X[is_used], y[is_used]


def getImageAsData(img_path, height, width, channels):
    """
        Return numpy array representing an image from
        given path scaled to given dimensions. If
        channels is 1 then image turned into grayscale.
    """
    try:
        # loads in as BGR not RGB:
        if channels > 1:
            img_arr = cv2.imread(img_path)
        else:  # for grayscale images:
            img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize now:
        """
            If you are enlarging the image, you should prefer to
            use INTER_LINEAR or INTER_CUBIC interpolation. If you are
            shrinking the image, you should prefer to use INTER_AREA.
            Cubic interpolation is computationally more complex,
            and hence slower than linear interpolation. However,
            the quality of the resulting image will be higher.
        """
        # plt.imshow(img_arr)  # img before resizing
        # plt.show()
        if width > img_arr.shape[0] or height > img_arr.shape[1]:
            # shrinking
            img_arr = cv2.resize(img_arr, (width, height),
                                interpolation=cv2.INTER_AREA)
        else:  # enlarging
            img_arr = cv2.resize(img_arr, (width, height),
                                interpolation=cv2.INTER_CUBIC)
        # img_rgb_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        # plt.imshow(img_rgb_arr)  # img after resizing
        # plt.show()
        return img_arr
    # in case of failure log a message and return None
    except Exception as e:
        logging.error("Failed to getImageAsData\nSkipped over Error:\n\t{}\n\n"
                      .format(e))
        return None


def storeNumpyArrays(np_arr, file_path, data_key="train_X"):
    """
        Stores given numpy array as either .npy file or
        .h5 depending on the extension in the provided file_path.
        In case of the .h5 file a key(data_key) needs to be
        provided and the data will be appended if .h5 file
        already exists with given path.
    """
    extension = file_path[file_path.rfind('.') : ]

    if extension == ".npy":
        np.save(file_path, np_arr)
        # to load back from file use:
        # train_X = np.load("train_X.npy")
    elif extension == ".h5":
        if os.path.isfile(file_path):
            msg = "Overwriting existing .h5 file:\n{}\n"\
                  .format(file_path)
            logging.warning(msg)
            os.remove(file_path)
        with h5py.File(file_path, 'a') as hdf5_file:
            hdf5_file.create_dataset(data_key, data=np_arr)
    else:
        logging.error("The format {} is not yet supported."
                      .format(extension))


# Loading in the created datasets:


def loadDataSets(path_dataset, seed=None, dataset_type=".npy"):
    """
        Loads in the datasets and shuffles them with provided
        postive seed or if seed is None sets random seed,
        if seed <= 0 then the data is not shuffled.
    """
    if dataset_type != ".npy" and  dataset_type != ".h5":
        msg = "{} dataset type is not supported (only .npy and .h5)\n" \
              .format(dataset_type)
        logging.error(msg)
        raise Exception(msg)

    train_set_X_orig = None
    train_set_y = None
    test_set_X_orig = None
    test_set_y = None
    classes = None

    path_X_train = os.path.join(path_dataset, "train_X.npy")
    path_y_train = os.path.join(path_dataset, "train_y.npy")
    path_X_test = os.path.join(path_dataset, "test_X.npy")
    path_y_test = os.path.join(path_dataset, "test_y.npy")

    # in case of .npy keys will be ignored
    data_key_X_train='train_set_x'
    data_key_y_train='train_set_y'
    data_key_X_test='test_set_x'
    data_key_y_test='test_set_y'

    train_set_X_orig, train_set_y, test_set_X_orig, \
    test_set_y, classes = loadImageDataSets(path_dataset, path_X_train,
                            path_y_train, path_X_test, path_y_test,
                            data_key_X_train=data_key_X_train,
                            data_key_y_train=data_key_y_train,
                            data_key_X_test=data_key_X_test,
                            data_key_y_test=data_key_y_test)

    # shuffle if seed was not specified or is > 0
    if seed == None:
        seed = np.random.randint(1, 1000)

    if seed > 0:
        logging.info("\nShuffling input data with seed: {}\n".format(seed))
        # so the shuffling would be the same for x, y do not
        # make 1 psrng object but initialize the same sequence over again
        # instead of just 1 sequence and using continuos/different parts of it
        np.random.RandomState(seed).shuffle(train_set_X_orig)
        np.random.RandomState(seed).shuffle(train_set_y)
        testing_seed = seed
        np.random.RandomState(testing_seed).shuffle(test_set_X_orig)
        np.random.RandomState(testing_seed).shuffle(test_set_y)

    return train_set_X_orig, train_set_y, test_set_X_orig, \
           test_set_y, classes


def loadImageDataSets(path_dataset, path_X_train, path_y_train,
                      path_X_test, path_y_test, data_key_X_train='',
                      data_key_y_train='', data_key_X_test='',
                      data_key_y_test=''):
    """
        Returns:
            train_set_X_orig, train_set_y, test_set_X_orig,
            test_set_y, classes
        Expects:
            categories.csv directly inside path_dataset folder.
            Mode that will indicate whether its .h5 or .npy.
            Given .npy or .h5 files to be stored in appropriate paths.
            In case of .h5 file the appropriate keys need to be provided.
    """

    # load in the datasets:
    train_set_X_orig = loadInNumpyArray(path_X_train, data_key=data_key_X_train)
    train_set_y = loadInNumpyArray(path_y_train, data_key=data_key_y_train)
    test_set_X_orig = loadInNumpyArray(path_X_test, data_key=data_key_X_test)
    test_set_y = loadInNumpyArray(path_y_test, data_key=data_key_y_test)
    classes = loadClasses(path_dataset)

    return train_set_X_orig, train_set_y, test_set_X_orig, test_set_y, classes


def loadInNumpyArray(file_path, data_key="train_X"):
    """
        Returns numpy array from .npy or .h5 file. In case of
        .h5 file a key needs to be provided.
        Raises custom exceptions on error.
    """
    np_arr = None
    extension = file_path[file_path.rfind('.') : ]

    if extension == ".npy":
        np_arr = np.load(file_path)
    elif extension == ".h5":
        with h5py.File(file_path, 'r') as hdf5_file:
            keys = hdf5_file.keys()
            if data_key not in keys:
                msg = "Provided key: {}; does not exist in the file: {};" \
                      .format(data_key, file_path)
                # logging.error(msg)
                raise DataHandlerException(msg, '')
            np_arr = np.array(hdf5_file.get(data_key))
    else:
        msg = "The format {} is not yet supported." \
              .format(extension)
        # logging.error(msg)
        raise DataHandlerException(msg, '')

    return np_arr


def loadClasses(path_dataset):
    """
        Returns python list of names of categories in
        path_dataset/categories.csv
        that can be used for example in classifciation problems.
    """
    categories = []
    with open(os.path.join(path_dataset, "categories.csv"),
              "r", encoding="utf-8") as file:
        categories = csv.reader(file, delimiter=",",
                                skipinitialspace=True).__next__()
    return categories


def getAllDirectoryImgsAsNumpyArr(path_arg, height, width,
                                  channels=3):
    """ Returns numpy array of images stored at locations
        described by path_arg.

        Height and width should be dimensions of images on
        which the model was trained.

        Path_arg can be a list of paths or a name of the
        directory where all the images are. If list then
        expects path_arg to be list of absolute paths to
        existing images.
    """
    paths = []

    if isinstance(path_arg, str):
        # path_arg is path to folder with imgs
        if not os.path.isdir(path_arg):
            raise Exception("Provided directory does not exist")

        for dirpath, dirnames, filenames in os.walk(path_arg):
            for img_name in filenames:
                cur_path = os.path.join(dirpath, img_name)
                paths.append(cur_path)
    else:  # in case path_arg is a list already
        paths = path_arg

    test_data = []
    successfully_loaded_imgs_paths = []
    for path in paths:
        image_data = getImageAsData(path, width, height,
                                    channels=channels)
        if image_data is None or not image_data.any():
            logging.debug("Skipped over img: {}".format(path))
            continue
        # successfully got the image data
        test_data.append(image_data)
        successfully_loaded_imgs_paths.append(path)

    test_data = np.array(test_data)

    # sanity check, view random img
    # img_idx = 0
    # print("Img sample #{}".format(img_idx))
    # img_rgb_arr = cv2.cvtColor(test_data[img_idx],
    #                            cv2.COLOR_BGR2RGB)
    # plt.imshow(img_rgb_arr)
    # plt.show()

    return test_data, successfully_loaded_imgs_paths


# Used insise the neural net:


def getRandomMiniBatches(X, Y, examples_axis, mini_batch_size=64, seed=None):
    """
        Returns shuffled copy of X and Y cut into mini batches of given size.
        If provided seed is negative does not shuffle.
        examples_axis - axis in X corresponding to data samples.
        Y should be of shape (num_labels, num_data_samples).
    """
    if seed is None:
        seed = np.random.randint(10000)
        logging.info("Randomly set seed for mini batches: {}".format(seed))

    num_training_examples = X.shape[examples_axis]
    mini_batches = []

    if seed >= 0:
        permutation = np.random.RandomState(seed).permutation(num_training_examples)
    else:
        permutation = range(num_training_examples)

    shuffled_X = np.take(X, permutation, axis=examples_axis)
    shuffled_Y = Y[: , permutation]

    num_complete_minibatches = num_training_examples // mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = np.take(shuffled_X,
                               range(k * mini_batch_size, (k+1) * mini_batch_size),
                               axis=examples_axis)
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # last mini batch < mini_batch_size
    if num_training_examples % mini_batch_size != 0:
        to_select = range( (num_training_examples // mini_batch_size) \
                          * mini_batch_size, num_training_examples )
        mini_batch_X = np.take(shuffled_X, to_select, axis=examples_axis)
        mini_batch_Y = shuffled_Y[:, (num_training_examples // mini_batch_size) * mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# Unused, but useful utility functions:


def getOneHotEncoded_Y(orig_Y, num_classes):
    """
        Returns a one hot encoded copy/version of given orig_Y.
        Expects orig_Y to be of shape (1, num_samples).
    """
    num_samples = orig_Y.shape[1]
    
    new_Y = np.zeros((num_classes, num_samples))
    for i in range(num_samples):
        new_Y[orig_Y[0, i]][i] = 1

    return new_Y


def getOneHotDecoded_Y(orig_Y,
                       binary_classification_positive_threshold=0.5):
    """
        Returns a one hot decoded copy/version of given orig_Y of
        shape (1, num_samples).
        Expects orig_Y to be of shape (num_classes, num_samples).
    """
    if orig_Y.shape[0] == 1:  # binary classification
        return (orig_Y >= binary_classification_positive_threshold)\
               .astype(np.uint8)

    return np.argmax(orig_Y, axis=0).reshape(1, -1)


# RUNNER
if __name__ == "__main__":
    # main()
    pass
