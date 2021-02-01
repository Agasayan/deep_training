import numpy as np
import pickle
import random

from keras.utils import to_categorical

# Dictionary for counting how many times each word has appeared in all the data.
# From given string, extract each word and check if it is added do dictionary.
# If not, add to dictionary. Otherwise increment number of occurrences of this word
def add_to_dict(dictionary, line):
    for word in line.split():
        just_letters_word = ''.join(e for e in word.lower() if e.isalnum())
        if len(just_letters_word) > 0 and just_letters_word not in dictionary:
            dictionary[just_letters_word] = 0
        elif len(just_letters_word) > 0:
            dictionary[just_letters_word] += 1
    return dictionary


# from sorted dictionary - shuffle keys and then iterate for all keys and set value i for i-th key
def one_hot_dictionary(dictionary):
    keys = list(dictionary.keys())
    random.shuffle(keys)
    dictionary = {x: y for y, x in enumerate(list(dictionary.keys()))}
    return dictionary


# From given dictionary, create sorted by values dictionary of given length
def create_sorted_dict(dictionary, length):
    # Get dictionary items, sort it, then cut redundant data if possible
    dict_items = list(dictionary.items())
    dict_items.sort(key=lambda x: x[1], reverse=True)
    if length < len(dict_items):
        dict_items = dict_items[:length]
    else:
        print("Provide correct length of dictionary")
        return 0
    sorted_dictionary = dict(dict_items)
    return sorted_dictionary


# Create dictionary (of given length) of  the most often occurring words
# At the end, save to pickle.
def create_dictionary(data_set, length):
    dictionary = {}
    for element in data_set:
        # from each example of data, get description and title to create dictionary
        description = np.array(element['description'], ndmin=1)
        title = np.array(element['title'], ndmin=1)

        # change to string and delete first character which is some random stuff
        description[0] = str(description[0])
        description[0] = description[0][1:]
        title[0] = str(title[0])
        title[0] = title[0][1:]
        dictionary = add_to_dict(dictionary, description[0])
        dictionary = add_to_dict(dictionary, title[0])
    # First create sorted dictionary. Then pass returned dictionary into function
    # which will shuffle keys and set values from 0 to length of dict
    dictionary = one_hot_dictionary(create_sorted_dict(dictionary, length))
    filename = "dictionary"
    outfile = open(filename, 'wb')
    pickle.dump(dictionary, outfile)
    outfile.close()
    # return dictionary


# using dictionary and numpy, create one-hot vectors of length of dictionary for train data and labels
def create_one_hot_vectors(dictionary, data_set, length, is_train_data=True):
    data_set = data_set.take(length)
    x_data = np.zeros(shape=(length, len(dictionary)))
    y_data = np.zeros(shape=(length, 4))
    counter = 0
    for element in data_set:

        # FIRST PART ---------------------------------------------------------------------------------------------------
        # Create one hot vectors for description + labels using one-hot dictionary
        description = np.array(element['description'], ndmin=1)
        title = np.array(element['title'], ndmin=1)
        description[0] = str(description[0])
        description[0] = description[0][1:]
        title[0] = str(title[0])
        title[0] = title[0][1:]
        description[0] += " " + title[0]
        # check if word in dictionary. if true, then set value 1 at index equal to value of this word in dictionary
        for word in description[0].split():
            if word in dictionary:
                x_data[counter][dictionary[word]] = 1

        # SECOND PART --------------------------------------------------------------------------------------------------
        # Create one hot labels. We have 4 classes in total in this data set
        label = np.array(element['label'], ndmin=1)
        tmp_arr = np.zeros(4)
        tmp_arr[label[0]] = 1
        # label = to_categorical(label)
        y_data[counter] = tmp_arr
        counter += 1
    # save data to pickle
    if is_train_data:  # default case is for train data
        filename_x_data = "train_data"
        filename_y_data = "train_labels"
        outfile_x_data = open(filename_x_data, 'wb')
        outfile_y_data = open(filename_y_data, 'wb')
        pickle.dump(x_data, outfile_x_data)
        outfile_x_data.close()
        pickle.dump(y_data, outfile_y_data)
        outfile_y_data.close()
    else:  # else is for test data
        filename_x_data = "test_data"
        filename_y_data = "test_labels"
        outfile_x_data = open(filename_x_data, 'wb')
        outfile_y_data = open(filename_y_data, 'wb')
        pickle.dump(x_data, outfile_x_data)
        outfile_x_data.close()
        pickle.dump(y_data, outfile_y_data)
        outfile_y_data.close()
    # return x_data, y_data
