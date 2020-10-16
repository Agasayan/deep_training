import csv
import numpy as np
import pandas as pd
from keras import models
from keras import layers
import matplotlib.pyplot as plt

test_data = [0]
with open('test_data.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONE)
    for row in reader:
        int_row = [0]
        for i in row:
            int_row.append(int(i))
        int_row.pop(0)
        test_data.append(int_row)
test_data.pop(0)

test_labels = [0]
with open('test_labels.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONE)
    for row in reader:
        int_row = [0]
        for i in row:
            int_row.append(int(i))
        int_row.pop(0)
        test_labels.append(int_row)
test_labels.pop(0)

train_data = [0]
with open('train_data.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONE)
    for row in reader:
        int_row = [0]
        for i in row:
            int_row.append(int(i))
        int_row.pop(0)
        train_data.append(int_row)
train_data.pop(0)

train_labels = [0]
with open('train_labels.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONE)
    for row in reader:
        int_row = [0]
        for i in row:
            int_row.append(int(i))
        int_row.pop(0)
        train_labels.append(int_row)
train_labels.pop(0)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
dt_train = pd.DataFrame(x_train)
dt_train_labels = pd.DataFrame(train_labels)
dt_train_full = pd.concat([dt_train, dt_train_labels], axis='columns').sample(frac=1).reset_index(drop=True)
x_train = dt_train_full.iloc[:, 0:-20].values
train_labels = dt_train_full.iloc[:, -20:]

x_test = vectorize_sequences(test_data)
dt_test = pd.DataFrame(x_test)
dt_test_labels = pd.DataFrame(test_labels)

dt_test_full = pd.concat([dt_test, dt_test_labels], axis='columns').sample(frac=1).reset_index(drop=True)
x_test = dt_test_full.iloc[:, 0:-20].values
test_labels = dt_test_full.iloc[:, -20:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=8, batch_size=512, validation_data=(x_val, y_val))

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.clf()
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

results = model.evaluate(x_test, test_labels)
print(results)
