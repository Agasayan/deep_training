from keras.datasets import cifar100
import tensorflow as tf
from keras import layers
from keras import models
from keras.utils import to_categorical
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.load_data()
        self.model = models.Sequential()
        self.classes = len(self.y_test[0])
        self.layer_activation = 'relu'
        self.output_activation = 'softmax'

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train / 255
        x_val = x_train[45000:]
        x_train = x_train[:45000]
        x_test = x_test / 255

        y_train = to_categorical(y_train)
        y_val = y_train[45000:]
        y_train = y_train[:45000]
        y_test = to_categorical(y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def build(self, optimizer, loss, metrics):
        self.model.add(layers.Conv2D(128, (3, 3), activation=self.layer_activation, input_shape=(32, 32, 3)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(128, (3, 3), activation=self.layer_activation))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(256, (3, 3), activation=self.layer_activation))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(256, (3, 3), activation=self.layer_activation))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation=self.layer_activation))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(100, activation=self.output_activation))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, epochs, batch_size):
        return self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                              validation_data=(self.x_val, self.y_val))

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("accuracy:", test_acc, "loss:", test_loss)


def graphs(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


Net = Model()
Net.load_data()
Net.build('Adam', 'categorical_crossentropy', ['accuracy'])
Net.model.summary()
history = Net.train(30, 128)
graphs(history)
Net.evaluate()
