import email.message
import pathlib
import os
import csv
from collections import Counter

PATH = "/home/hubert/Datasets/"
TRAIN_DIR = "20news-bydate-train/"
TEST_DIR = "20news-bydate-test/"
PATH_TO_TRAIN_DATA = PATH + TRAIN_DIR
PATH_TO_TEST_DATA = PATH + TEST_DIR
directories = os.listdir(PATH_TO_TRAIN_DATA)


def mail_to_words(mail):  # creating list of words from mail
    only_letters_mail = [0]
    for word in mail.split():  # split string into list of words
        only_letters_word = ""
        for char in word:  # delete all special characters
            if char.isalpha():
                only_letters_word += char.lower()
            else:
                continue
        if only_letters_word != "":
            only_letters_mail.append(only_letters_word)
    only_letters_mail.pop(0)
    return only_letters_mail


def most_frequent_words(list, n):  # create list of n most frequent words from list
    occurrence_count = Counter(list)  # using Counter and most_common to get the words
    list_of_most_frequent = occurrence_count.most_common(n)
    only_words = [0]  # create list only with words, without frequency
    for i in list_of_most_frequent:
        only_words.append(i[0])
    only_words.pop(0)
    return only_words


def create_data_dictionary(directory, n):  # create list of n most frequent words in all the data
    all_words = [0]
    # add words from the train data
    for i in directory:  # iterate through all directories
        sub_dir = PATH_TO_TRAIN_DATA + i
        for path in pathlib.Path(sub_dir).iterdir():  # iterate through all files in directory
            if path.is_file():
                try:
                    file = open(path, "r")
                    mail = email.message_from_file(file)
                    all_words += mail_to_words(mail.as_string())
                except UnicodeDecodeError:
                    os.remove(path)
                    continue
    # add words from the test data
    for i in directory:
        sub_dir = PATH_TO_TEST_DATA + i
        for path in pathlib.Path(sub_dir).iterdir():
            if path.is_file():
                try:
                    file = open(path, "r")
                    mail = email.message_from_file(file)
                    all_words += mail_to_words(mail.as_string())
                except UnicodeDecodeError:
                    os.remove(path)
                    continue
    all_words.pop(0)
    return most_frequent_words(all_words, n)


def create_vector(message, dictionary):  # create vector of dictionary-only words from given mail
    data_vector = [0]
    for word in message:
        for i in range(
                len(dictionary)):  # go through the dictionary and add 'i' to vector if current word is listed in it
            if word == dictionary[i]:
                data_vector.append(i)
                break
    data_vector.pop(0)
    return data_vector


def vectorize_data(directory, dictionary):
    # TRAIN DATA ------------------------------------------------------------------------------------------
    label_counter = 0
    train_labels = [0]
    train_data_vector = [0]
    for i in directory:  # first, translate mails into vectors of dictionary-only words
        sub_dir = PATH_TO_TRAIN_DATA + i
        for path in pathlib.Path(sub_dir).iterdir():
            file = open(path, "r")
            mail = email.message_from_file(file)

            # translate mail to string, then letters-only strings and vector at last
            vector = create_vector(mail_to_words(mail.as_string()), dictionary)
            train_data_vector.append(vector)  # add vectorized data

            label = [0] * 20  # create label
            label[label_counter] = 1
            train_labels.append(label)  # add to labels vector
        label_counter += 1  # next directory means next emails topic
    train_data_vector.pop(0)
    train_labels.pop(0)
    # TEST DATA ------------------------------------------------------------------------------------------
    label_counter = 0
    test_labels = [0]
    test_data_vector = [0]
    for i in directory:  # first, translate mails into vectors of dictionary-only words
        sub_dir = PATH_TO_TEST_DATA + i
        for path in pathlib.Path(sub_dir).iterdir():
            file = open(path, "r")
            mail = email.message_from_file(file)

            # translate mail to string, then letters only strings and vectors at last
            vector = create_vector(mail_to_words(mail.as_string()), dictionary)
            test_data_vector.append(vector)  # add vectorized data to main vector

            label = [0] * 20  # create label
            label[label_counter] = 1
            test_labels.append(label)  # add to labels vector
        label_counter += 1  # next directory means next emails topic
    test_data_vector.pop(0)
    test_labels.pop(0)

    return train_data_vector, train_labels, test_data_vector, test_labels


data_dictionary = create_data_dictionary(directories, 10000)
(train_data, train_labels, test_data, test_labels) = vectorize_data(directories, data_dictionary)

with open('train_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in train_data:
        writer.writerow(i)
with open('train_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in train_labels:
        writer.writerow(i)
with open('test_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in test_data:
        writer.writerow(i)
with open('test_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in test_labels:
        writer.writerow(i)
