import keras
import keras.backend as K
from keras.layers import Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Dense
# import tensorflow.keras.preprocessing.image as img

import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt

# import seaborn as sns

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=config))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_images = train_images / 255.0
test_images = test_images / 255.0
# training_data_generator = img.ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=25,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     rescale=1. / 255)
# test_data_generator = img.ImageDataGenerator(rescale=1. / 255)
#
# train_generator = training_data_generator.flow(train_images, train_labels, batch_size=96)
# test_generator = test_data_generator.flow(test_images, test_labels, batch_size=96)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1), name="Flatten_1"),
    Reshape([28, 28, 1], name="Reshape"),
    BatchNormalization(name='BatchNorm_1'),
    Conv2D(32, (3, 3), padding='same', activation='relu', name='Conv2D_1'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),
    Conv2D(64, (3, 3), activation='relu', name='Conv2D_2'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu', name='dense_1'),
    Dropout(0.5),
    Dense(64, activation='relu', name='dense_2'),
    BatchNormalization(name='BatchNorm_2'),
    Dense(10, activation='softmax', name='dense_last')
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
]

# model_history = model.fit_generator(train_generator, epochs=100, validation_data=test_generator, callbacks=callbacks)
model_history = model.fit(train_images, train_labels, batch_size=96, epochs=8, callbacks=callbacks,
                          validation_data=(test_images, test_labels))


def plot_history(history, key):
    plt.figure(figsize=(16, 10))
    name = 'model_history'
    val = plt.plot(history.epoch, history.history['val_' + key],
                   '--', label=name.title() + ' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()


plot_history(model_history, 'loss')

model.save('fashionMNIST.h5')

# # Create dictionary of target classes
# label_dict = {
#     0: 'T-shirt/top',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
# }