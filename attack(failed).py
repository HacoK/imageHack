import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

import numpy as np
import matplotlib.pyplot as plt

trained_model = keras.models.load_model('fashionMNIST.h5')

# Disassemble layers
layerL = [l for l in trained_model.layers]

original = tf.reshape(layerL[0].input, [-1, 784])
max_change_above = original + 0.005
max_change_below = original - 0.005
max_change_above = tf.cast(max_change_above, tf.float32)
max_change_below = tf.cast(max_change_below, tf.float32)


def clip(embed1, min, max):
    embed = tf.clip_by_value(embed1, min, max)
    return embed


start_layer_model = models.Model(inputs=trained_model.input, outputs=trained_model.get_layer('Flatten_1').output)

modify_1 = layers.Dense(784, activation='relu', name='modify_1')(start_layer_model.output)
modify_2 = layers.Dense(784, activation='relu', name='modify_2')(modify_1)
modify_3 = layers.Dense(784, activation='relu', name='modify_3')(modify_2)
clip_3_1 = layers.Lambda(clip, arguments={'min': max_change_below, 'max': max_change_above})(modify_3)
clip_3_2 = layers.Lambda(clip, arguments={'min': 0, 'max': 1}, name='Gen')(clip_3_1)

x = clip_3_2

for i in range(1, len(layerL)):
    x = layerL[i](x)
    layerL[i].trainable = False

symmetric_model = models.Model(inputs=layerL[0].input, outputs=x)
generator_model = models.Model(inputs=symmetric_model.input, outputs=symmetric_model.get_layer('Gen').output)

# custom loss
def mycrossentropy(y_true, y_pred):
    return -keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


symmetric_model.compile(optimizer=keras.optimizers.Adam(),
                        loss=mycrossentropy,
                        metrics=['accuracy'])
# symmetric_model.summary()

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=config))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_images = train_images / 255.0
test_images = test_images / 255.0

callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
]
# model_history = model.fit_generator(train_generator, epochs=100, validation_data=test_generator, callbacks=callbacks)
model_history = symmetric_model.fit(train_images, train_labels, batch_size=96, epochs=8, callbacks=callbacks,
                                    validation_data=(test_images, test_labels))


def display_digit(images, labels, num):
    print(labels[num])
    plt.figure(figsize=(6, 3))
    label = labels[num].argmax(axis=0)
    image = images[num].reshape([28, 28])
    craft = np.reshape(generator_model.predict(np.expand_dims(images[num], 0)), (28, 28))
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.subplot(1, 2, 2)
    plt.imshow(craft * 255, cmap=plt.get_cmap('gray_r'))
    plt.show()


display_digit(test_images, test_labels, 89)

print(np.argmax(trained_model.predict(np.expand_dims(test_images[89], 0))))
print(np.argmax(
    trained_model.predict(np.reshape(generator_model.predict(np.expand_dims(test_images[89], 0)), (1, 28, 28, 1)))))