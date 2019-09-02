import numpy as np
import keras

from util import KerasModel, AdditiveUniformNoiseAttack
from util.criteria import OriginalClassProbability, Misclassification, ConfidentMisclassification


def aiTest(images, shape):
    # # Define the characteristics of the input image
    # imageNum = len(images)
    # width = shape[1]
    # height = shape[2]
    # channels = shape[3]

    # if channels == 3:
    #     for i in range(imageNum):
    #         r = images[i, :, :, 0]
    #         g = images[i, :, :, 1]
    #         b = images[i, :, :, 2]
    #         images[i] = cv2.merge([r, g, b])
    # if (width != 28) or (height != 28):
    #     width = 28
    #     height = 28

    images = images / 255.0
    trained_model = keras.models.load_model('fashionMNIST.h5')
    cnn5 = keras.models.load_model('cnn5.h5')
    sub_model = KerasModel(trained_model, bounds=(0, 1), channel_axis=1)

    generate_images = []
    labels = np.argmax(cnn5.predict(images), axis=1)

    for i in range(shape[0]):
        # if i % 50 == 0:
        #     print("{:3d}".format(i // 50), end=": ")
        # print(i % 50, end=" ")
        # if i % 50 == 49:
        #     print()
        image = images[i]
        label = labels[i]
        generate = None
        if generate is None:
            attack = AdditiveUniformNoiseAttack(model=sub_model,
                                                criterion=OriginalClassProbability(0.1))
            generate = attack(image, label=label)
        if generate is None:
            attack = AdditiveUniformNoiseAttack(model=sub_model,
                                                criterion=ConfidentMisclassification(0.9))
            generate = attack(image, label=label)
        if generate is None:
            attack = AdditiveUniformNoiseAttack(model=sub_model,
                                                criterion=Misclassification())
            attack(image, label=label)
        if generate is None:
            randIndex = np.random.randint(0, shape[0])
            generate = images[randIndex]
        generate_images.append(generate)

    generate_images = np.array(generate_images)
    generate_images = generate_images * 255
    return generate_images

# # self_test
# import time
# import skimage
#
# (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
# train_images = np.expand_dims(train_images, axis=3)
# test_images = np.expand_dims(test_images, axis=3)
#
# testNum = 100
# randArray = np.random.choice(10000, testNum, replace=False)
# testImages = []
# for i in randArray:
#     testImages.append(test_images[i])
#
# startTime = time.time()
# generate_images = aiTest(np.array(testImages), (testNum, 28, 28, 1))
# endTime = time.time() - startTime
#
# cnn5 = keras.models.load_model('cnn5.h5')
# trained_model = keras.models.load_model('fashionMNIST.h5')
# attackRate = 0.0
# attackSSIM = 0.0
#
# for i in range(testNum):
#     origin_label = np.argmax(cnn5.predict(np.expand_dims(testImages[i] / 255.0, 0))[0])
#     fake_label = np.argmax(cnn5.predict(np.expand_dims(generate_images[i] / 255.0, 0))[0])
#     if origin_label != fake_label:
#         attackRate = attackRate + 1
#         attackSSIM = attackSSIM + skimage.measure.compare_ssim(testImages[i].reshape([28, 28]),
#                                                                generate_images[i].reshape([28, 28]))
#
# attackSSIM = attackSSIM / attackRate
# attackRate = attackRate / testNum
#
# print("----------------------------------------------------------------------")
# print("attackRate:", attackRate)
# print("attackSSIM:", attackSSIM)
# print("endTime:", endTime, "sec")
