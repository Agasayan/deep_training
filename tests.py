# # train_ds = tfds.load('ag_news_subset', split='train', shuffle_files=True)
# # assert isinstance(train_ds, tf.data.Dataset)
#
#
# # train_ds = train_ds.take(30000)
#
#
# # description = np.array(train_ds['description'])
# # print(description)
#
# def prepare_data(input_data):
#     x1_train = [" "] * 120000
#     x2_train = [" "] * 120000
#     y_train = [[]] * 120000
#     counter = 0
#     for example in input_data:
#         new_description = ''
#         description = np.array(example['description'], ndmin=1)
#         label = np.array(example['label'], ndmin=1)
#         new_label = to_categorical(label, 4)
#         y_train[counter] = new_label
#         new_title = ''
#         title = np.array(example['title'], ndmin=1)
#         description[0] = str(description[0])
#         description[0] = description[0][1:]
#         title[0] = str(title[0])
#         title[0] = title[0][1:]
#
#         for word in description[0].split():
#             better_word = ''.join(e for e in word.lower() if e.isalnum())
#
#             new_description += better_word + ' '
#
#             # if len(better_word) > 0:
#             #     occurence_array[dict.get(better_word) - 1] += 1
#
#         for word in title[0].split():
#             better_word = ''.join(e for e in word.lower() if e.isalnum())
#
#             new_title += better_word + ' '
#             # if len(better_word) > 0:
#             #     occurence_array[dict.get(better_word) - 1] += 1
#
#         x1_train[counter] += new_description
#         x2_train[counter] += new_title
#         counter += 1
#         print(counter)
#     for i in range(len(x1_train)):
#         x1_train[i] += " " + x2_train[i]
#     return x1_train, y_train
#
#     # return occurence_array
#
#
# def create_dictionary(input_data):
#     token_index = {}
#     for example in input_data:
#         new_description = ''
#         description = np.array(example['description'], ndmin=1)
#
#         new_title = ''
#         title = np.array(example['title'], ndmin=1)
#         description[0] = str(description[0])
#         description[0] = description[0][1:]
#         title[0] = str(title[0])
#         title[0] = title[0][1:]
#
#         for word in description[0].split():
#             better_word = ''.join(e for e in word.lower() if e.isalnum())
#             new_description += better_word + ' '
#             if len(better_word) > 0 and better_word not in token_index:
#                 token_index[better_word] = 0
#             elif len(better_word) > 0:
#                 token_index[better_word] += 1
#
#         for word in title[0].split():
#             better_word = ''.join(e for e in word.lower() if e.isalnum())
#             new_title += better_word + ' '
#             if len(better_word) > 0 and better_word not in token_index:
#                 token_index[better_word] = 0
#             elif len(better_word) > 0:
#                 token_index[better_word] += 1
#     return token_index
#
#
# def create_one_hot_vectors_of_data(input_data, dictionary):
#     data_vectorized = np.zeros(shape=(len(input_data), len(dictionary)))
#     for n in range(len(input_data)):
#         for word in input_data[n].split():
#             data_vectorized[n][dictionary[word]] = 1
#     return data_vectorized
#
#
# def cut_data_into8(data):
#     n = len(data) / 8
#     n = int(n)
#     return data[:n], data[n:2 * n], data[2 * n:3 * n], data[3 * n:4 * n], data[4 * n:5 * n], data[5 * n:6 * n], \
#            data[6 * n:7 * n], data[7 * n:8 * n]
#
#     # dict_of_occurrence = {}
#     # exit(0)
#     # keys = list(dictionary.keys())
#     # random.shuffle(keys)
#     # shuffled_dictionary = dict(zip(keys, dictionary.values()))
#     # exit(0)
#     # filename = 'dictionary_10k_words'
#     # outfile = open(filename, 'wb')
#     # pickle.dump(dict_10k, outfile)
#     # outfile.close()
#     #
#     # train_data_vectorized = create_one_hot_vectors_of_data(x_train, shuffled_dictionary)
#     # filename = 'x_train'
#     # filename2 = 'y_train'
#     # outfile = open(filename, 'wb')
#     # pickle.dump(train_data_vectorized, outfile)
#     # outfile.close()
#     # outfile = open(filename2, 'wb')
#     # pickle.dump(y_train, outfile)
#     # outfile.close()
#     # infile = open('dictionary_10k_words', 'rb')
#     # dictionary = pickle.load(infile)
#     # infile.close()
#     # arr = create_one_hot_vectors_of_data(train_ds, dictionary)
#     # exit(0)
#
#
# # filename = 'x_test'
# # # filename2 = 'y_test'
# # infile1 = open(filename, 'rb')
# # # infile2 = open(filename2, 'rb')
# # x_test = pickle.load(infile1)
# # infile1.close()
#
#
# # y_train = pickle.load(infile2)
# # infile2.close()
# # x = (a, b, c, d, e, f, g, h) = cut_data_into8(x_train)
# # counter = 0
# # for cut in x:
# #     filename = "cut" + "_" + str(counter)
# #     outfile = open(filename, 'wb')
# #     pickle.dump(cut, outfile)
# #     outfile.close()
# #     counter += 1
# def cut_to_vector(data, dictionary, k):
#     vector = np.zeros(shape=(len(data), len(dictionary)))
#     for i in range(len(data)):
#         for word in data[i].split():
#             if word in dictionary:
#                 vector[i][dictionary[word]] = 1
#     filename = "train_" + str(k)
#     outfile = open(filename, 'wb')
#     pickle.dump(vector, outfile)
#     outfile.close()
#
#
# # x_test, y_test = prepare_data(train_ds)
#
# # filename = "one_hot_test_1"
# # infile = open(filename, 'rb')
# # data = pickle.load(infile)
# # infile.close()
# # data = data[:10000]
# #
# # filename = "dictionary_10k_words"
# # infile = open(filename, 'rb')
# # dictionary = pickle.load(infile)
# # infile.close()
# # items = list(dictionary.items())
# # items = items[:10000]
# # new_dictionary = dict(items)
# #
# # cut_to_vector(data, dictionary, 0)
# # exit(0)
#
#
# def one_hot_data(n):
#     filename = "dictionary_30k_words"
#     infile = open(filename, 'rb')
#     dictionary = pickle.load(infile)
#     infile.close()
#     for i in range(n):
#         filename = "cut_" + str(i)
#         infile = open(filename, 'rb')
#         cut = pickle.load(infile)
#         infile.close()
#         cut_to_vector(cut, dictionary, i)
#
#
# def save_labels(labels):
#     x = (a, b, c, d, e, f, g, h) = cut_data_into8(labels)
#     counter = 0
#     for cut in x:
#         filename = "labels_" + str(counter)
#         outfile = open(filename, 'wb')
#         pickle.dump(cut, outfile)
#         outfile.close()
#         counter += 1
#
# filename = "one_hot_test_2"
# infile = open(filename, 'rb')
# x_test = pickle.load(infile)
# infile.close()
# # x_test = np.asarray(x_test).resize((7600, 10000))
# # x_test = list(x_test)
# # for i in range(len(x_test)):
# #     x_test[i] = list(x_test[i])
# #     x_test[i] = x_test[i][:10000]
# #     x_test[i] = np.asarray(x_test[i])
# # x_test = np.asarray(x_test).reshape(len(x_test), len(x_test[0]))
# # filename = "one_hot_test_2"
# # outfile = open(filename, 'wb')
# # pickle.dump(x_test, outfile)
# # outfile.close()
# # exit(0)
# filename = "test_labels"
# infile = open(filename, 'rb')
# y_test = pickle.load(infile)
# y_test = np.asarray(y_test).reshape(len(y_test), 4)
# infile.close()

# a = [(1, 2), (3, 4), (5, 6)]
# a.sort(key=lambda x: x[1], reverse=True)
# print(len(a))
