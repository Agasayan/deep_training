import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# import tensorflow_datasets as tfds #UNCOMMENT FOR FIRST-TIME COMPILATION ---------------------------------------------
import pickle
import sys

sys.path.insert(0, "/home/hubert/PycharmProjects/Deep_learning")
from ag_news_subset.model import Model
from ag_news_subset.data_preparation import *

# ----------------------------------------------------------------------------------------------------------------------
# UNCOMMENT WHEN COMPILING FOR THE FIRST TIME
# ----------------------------------------------------------------------------------------------------------------------
# train_ds = tfds.load('ag_news_subset', split='train', shuffle_files=True)
# test_ds = tfds.load('ag_news_subset', split='test', shuffle_files=True)
# assert isinstance(train_ds, tf.data.Dataset)
# ----------------------------------------------------------------------------------------------------------------------
LENGTH = 12000
LENGTH_OF_DICT = 10000
TEST_LENGTH = 7600


# Function to open files for training and testing, and to create validate data
def read_data():
    filename_x_train = "train_data"
    filename_y_train = "train_labels"
    infile_x_data = open(filename_x_train, 'rb')
    infile_y_data = open(filename_y_train, 'rb')
    x_train = pickle.load(infile_x_data)
    infile_x_data.close()
    y_train = pickle.load(infile_y_data)
    infile_y_data.close()

    x_val = x_train[LENGTH - 2000:LENGTH]
    x_train = x_train[:LENGTH - 2000]

    y_val = y_train[LENGTH - 2000:LENGTH]
    y_train = y_train[:LENGTH - 2000]

    filename_x_test = "test_data"
    filename_y_test = "test_labels"
    infile_x_data = open(filename_x_test, 'rb')
    infile_y_data = open(filename_y_test, 'rb')
    x_test = pickle.load(infile_x_data)
    infile_x_data.close()
    y_test = pickle.load(infile_y_data)
    infile_y_data.close()
    return x_train, y_train, x_val, y_val, x_test, y_test


# ----------------------------------------------------------------------------------------------------------------------
# THE STUFF BELOW IS TO PREPARE DATA, WHEN COMPILING FOR THE FIRST TIME
# ----------------------------------------------------------------------------------------------------------------------

# create_dictionary(train_ds, LENGTH_OF_DICT)
# filename = 'dictionary'
# infile = open(filename, 'rb')
# dictionary = pickle.load(infile)
# infile.close()

# create_one_hot_vectors(dictionary, train_ds, LENGTH)
# create_one_hot_vectors(dictionary, test_ds, TEST_LENGTH, False)

# ----------------------------------------------------------------------------------------------------------------------
# END OF FIRST-COMPILATION STUFF
# ----------------------------------------------------------------------------------------------------------------------

x_train, y_train, x_val, y_val, x_test, y_test = read_data()

graph = Model(LENGTH_OF_DICT)
n_epochs = 30
# graph.train_with_validation(x_train, y_train, x_val, y_val, n_epochs)
graph.train(x_train, y_train, n_epochs)
test_loss, test_acc = graph.model.evaluate(x_test, y_test)
print(test_acc)
