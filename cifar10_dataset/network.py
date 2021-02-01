#koncowe acc = 78% przy 50 epokach i 1/2 danych
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.utils import to_categorical

N_EPOCHS = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_val = x_train[40000:45000] / 255
x_train = x_train[:20000] / 255
x_test = x_test / 255

y_val = to_categorical(y_train[40000:45000])
y_train = to_categorical(y_train[:20000])
y_test = to_categorical(y_test)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
# model.add(layers.MaxPooling2D((2, 2)))
# model.summary()
# exit(0)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))
# model.summary()
# exit(0)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=64, validation_data=(x_val, y_val))
print(history.history.keys())
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, N_EPOCHS + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)
