import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import pickle
from keras.utils import to_categorical

train_ds = tfds.load('ag_news_subset', split='train', shuffle_files=True)
assert isinstance(train_ds, tf.data.Dataset)


# train_ds = train_ds.take(10)


# description = np.array(train_ds['description'])
# print(description)

def create_dictionary(input_data):
    token_index = {}
    x1_train = []
    x2_train = []
    y_train = []
    for example in input_data:
        new_description = ''
        description = np.array(example['description'], ndmin=1)
        label = np.array(example['label'], ndmin=1)
        new_label = to_categorical(label, 4)
        y_train.append(new_label)
        new_title = ''
        title = np.array(example['title'], ndmin=1)
        description[0] = str(description[0])
        description[0] = description[0][1:]
        title[0] = str(title[0])
        title[0] = title[0][1:]

        for word in description[0].split():
            better_word = ''.join(e for e in word.lower() if e.isalnum())
            new_description += better_word + ' '
            if len(better_word) > 0 and better_word not in token_index:
                token_index[better_word] = len(token_index) + 1

        for word in title[0].split():
            better_word = ''.join(e for e in word.lower() if e.isalnum())
            new_title += better_word + ' '
            if len(better_word) > 0 and better_word not in token_index:
                token_index[better_word] = len(token_index) + 1
        x1_train.append(new_description)
        x2_train.append(new_title)
    for i in range(len(x1_train)):
        x1_train[i] += " " + x2_train[i]
    return token_index, x1_train, y_train


def create_one_hot_vectors_of_data(input_data, dictionaryy):
    results = np.zeros(shape=(len(input_data), 102169))
    for n in range(len(input_data)):
        for word in input_data[n].split():
            x = dictionaryy[word] - 1
            results[n][dictionaryy[word] - 1] = 1
    return results


dictionary, x_train, y_train = create_dictionary(train_ds)
keys = list(dictionary.keys())
random.shuffle(keys)
shuffled_dictionary = dict(zip(keys, dictionary.values()))
# filename = 'dictionary'
# outfile = open(filename, 'wb')
# pickle.dump(shuffled_dictionary, outfile)
# outfile.close()

train_data_vectorized = create_one_hot_vectors_of_data(x_train, shuffled_dictionary)
filename = 'x_train'
filename2 = 'y_train'
outfile = open(filename, 'wb')
pickle.dump(train_data_vectorized, outfile)
outfile.close()
outfile = open(filename2, 'wb')
pickle.dump(y_train, outfile)
outfile.close()
