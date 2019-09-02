import keras
import numpy as np
import time

from foolbox.models import KerasModel
from foolbox.attacks import AdditiveUniformNoiseAttack
from foolbox.criteria import OriginalClassProbability, Misclassification, ConfidentMisclassification

from skimage.measure import compare_ssim as ssim

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_images = train_images / 255.0
test_images = test_images / 255.0

trained_model = keras.models.load_model('fashionMNIST.h5')
cnn5 = keras.models.load_model('cnn5.h5')
sub_model = KerasModel(trained_model, bounds=(0, 1), channel_axis=1)

testNum = 100
randArray = np.random.choice(10000, testNum, replace=False)


def diff_ssim(img1, img2):
    img1 = (img1 * 255).reshape([28, 28])
    img2 = (img2 * 255).reshape([28, 28])
    return ssim(img1, img2)


attackRate_u2 = 0.0
attackSSIM_u2 = 0.0
count = 0

startTime = time.time()
for randIndex in randArray:
    origin_output = test_labels[randIndex]

    attack_u2 = AdditiveUniformNoiseAttack(model=sub_model,
                                           criterion=OriginalClassProbability(0.1))

    generate_u2 = attack_u2(test_images[randIndex], label=test_labels[randIndex])
    if generate_u2 is None:
        attack_u2 = AdditiveUniformNoiseAttack(model=sub_model,
                                               criterion=ConfidentMisclassification(0.9))
        generate_u2 = attack_u2(test_images[randIndex], label=test_labels[randIndex])
    if generate_u2 is None:
        attack_u2 = AdditiveUniformNoiseAttack(model=sub_model,
                                               criterion=Misclassification())
        generate_u2 = attack_u2(test_images[randIndex], label=test_labels[randIndex])
    if generate_u2 is None:
        randItem = np.random.randint(0, 60000)
        generate_u2 = train_images[randIndex]

    modify_output_u2 = trained_model.predict(generate_u2[np.newaxis, :, :, :])

    cnn5_output_u2 = cnn5.predict(generate_u2[np.newaxis, :, :, :])

    if (np.argmax(origin_output) != np.argmax(modify_output_u2)) and (
            np.argmax(origin_output) != np.argmax(cnn5_output_u2)):
        attackRate_u2 = attackRate_u2 + 1
        attackSSIM_u2 = attackSSIM_u2 + diff_ssim(test_images[randIndex], generate_u2)
        count = count + 1

endTime = time.time() - startTime

attackRate_u2 = attackRate_u2 / testNum
attackSSIM_u2 = attackSSIM_u2 / count

print("----------------------------------------------------------------------")
print("attackRate_u2:", attackRate_u2)
print("attackSSIM_u2:", attackSSIM_u2)
print("endTime:", endTime)
