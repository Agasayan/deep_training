from keras import models
from keras import layers
import tensorflow as tf


class Model:
    # pass length of data set as an only argument
    # rest of the parameters define by yourself
    def __init__(self, size_y):
        self.model = models.Sequential()
        self.size_y = size_y
        self.add_layers()
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # define layers here:
    def add_layers(self):
        self.model.add(layers.Dense(256, activation='relu', input_shape=(self.size_y,)))
        self.model.add(layers.Dropout(0.8))

        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.8))

        self.model.add(layers.Dense(128, activation='relu', input_shape=(self.size_y,)))
        self.model.add(layers.Dropout(0.6))

        self.model.add(layers.Dense(4, activation='softmax'))

    # train model with validation data
    def train_with_validation(self, train_set, train_labels, validate_set, validate_labels, epochs):
        history = self.model.fit(train_set, train_labels, validation_data=(validate_set, validate_labels),
                                 epochs=epochs,
                                 batch_size=256)
        return history

    # train model without validation
    def train(self, train_set, train_labels, epochs):
        history = self.model.fit(train_set, train_labels, epochs=epochs, batch_size=512)
        return history
