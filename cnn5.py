import copy

import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# download dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# label names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# pre-process
train_images = train_images / 255.0
test_images = test_images / 255.0

cnn5 = Sequential()
cnn5.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
cnn5.add(BatchNormalization())

cnn5.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
cnn5.add(BatchNormalization())
cnn5.add(MaxPooling2D(pool_size=(2, 2)))
cnn5.add(Dropout(0.25))

cnn5.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn5.add(BatchNormalization())
cnn5.add(Dropout(0.25))

cnn5.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn5.add(BatchNormalization())
cnn5.add(Dropout(0.25))

cnn5.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
cnn5.add(BatchNormalization())
cnn5.add(MaxPooling2D(pool_size=(2, 2)))
cnn5.add(Dropout(0.25))

cnn5.add(Flatten())

cnn5.add(Dense(512, activation='relu'))
cnn5.add(BatchNormalization())
cnn5.add(Dropout(0.5))

cnn5.add(Dense(128, activation='relu'))
cnn5.add(BatchNormalization())
cnn5.add(Dropout(0.5))

cnn5.add(Dense(64, activation='relu'))
cnn5.add(BatchNormalization())
cnn5.add(Dropout(0.25))

cnn5.add(Dense(10, activation='softmax'))

cnn5.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
# pre-process

cnn_train_images = copy.copy(train_images)
print(cnn_train_images.shape)
cnn_train_images = np.expand_dims(cnn_train_images, 3)
print(cnn_train_images.shape)

history5 = cnn5.fit(cnn_train_images, train_labels, batch_size=256, epochs=10, verbose=1)

cnn_test_images = copy.copy(test_images)
print(cnn_test_images.shape)
cnn_test_images = np.expand_dims(test_images, 3)
print(cnn_test_images.shape)

score5 = cnn5.evaluate(cnn_test_images, test_labels, verbose=0)
print('Test loss:', score5[0])
print('Test accuracy:', score5[1])

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(cnn_train_images, train_labels, batch_size=256)

history5 = cnn5.fit_generator(batches, steps_per_epoch=48000 // 256, epochs=50)

score5 = cnn5.evaluate(cnn_test_images, test_labels, verbose=0)
print('Test loss:', score5[0])
print('Test accuracy:', score5[1])

cnn5.save("cnn5.h5")
