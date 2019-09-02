import keras
import numpy as np
import matplotlib.pyplot as plt

from util import KerasModel,AdditiveUniformNoiseAttack
from util.criteria import Misclassification, OriginalClassProbability

from skimage.measure import compare_ssim as ssim

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_images = train_images / 255.0
test_images = test_images / 255.0

trained_model = keras.models.load_model('fashionMNIST.h5')
cnn5 = keras.models.load_model('cnn5.h5')

randIndex = np.random.randint(0, 60000)
origin_output = cnn5.predict(train_images[randIndex][np.newaxis, :, :, :])

print(np.argmax(origin_output))
print(origin_output)

sub_model = KerasModel(trained_model, bounds=(0, 1), channel_axis=1)

# attack_p = PointwiseAttack(model=sub_model,
#                            criterion=OriginalClassProbability(0.01))

# attack_u1 = AdditiveUniformNoiseAttack(model=sub_model,
#                                        criterion=Misclassification())
attack_u2 = AdditiveUniformNoiseAttack(model=sub_model,
                                       criterion=OriginalClassProbability(0.1))

# attack_g1 = AdditiveGaussianNoiseAttack(model=sub_model,
#                                         criterion=Misclassification())
# attack_g2 = AdditiveGaussianNoiseAttack(model=sub_model,
#                                         criterion=OriginalClassProbability(0.1))

# generate_p = attack_p(train_images[randIndex], label=train_labels[randIndex])
# generate_u1 = attack_u1(train_images[randIndex], label=train_labels[randIndex])
generate_u2 = attack_u2(train_images[randIndex], label=train_labels[randIndex])
# generate_g1 = attack_g1(train_images[randIndex], label=train_labels[randIndex])
# generate_g2 = attack_g2(train_images[randIndex], label=train_labels[randIndex])

# if generate_p is None:
#     print("NoneType")
#     exit(1)


def diff_ssim(img1, img2):
    img1 = (img1 * 255).reshape([28, 28])
    img2 = (img2 * 255).reshape([28, 28])
    return ssim(img1, img2)


# modify_output_p = trained_model.predict(generate_p[np.newaxis, :, :, :])
# modify_output_u1 = trained_model.predict(generate_u1[np.newaxis, :, :, :])
modify_output_u2 = trained_model.predict(generate_u2[np.newaxis, :, :, :])
# modify_output_g1 = trained_model.predict(generate_g1[np.newaxis, :, :, :])
# modify_output_g2 = trained_model.predict(generate_g2[np.newaxis, :, :, :])

# cnn5_output_p = cnn5.predict(generate_p[np.newaxis, :, :, :])
# cnn5_output_u1 = cnn5.predict(generate_u1[np.newaxis, :, :, :])
cnn5_output_u2 = cnn5.predict(generate_u2[np.newaxis, :, :, :])
# cnn5_output_g1 = cnn5.predict(generate_g1[np.newaxis, :, :, :])
# cnn5_output_g2 = cnn5.predict(generate_g2[np.newaxis, :, :, :])

# print("----------------------------------------------------------------------")
# print("modify_output_p:")
# print(np.argmax(modify_output_p))
# print(modify_output_p)
# print("cnn5_output_p:")
# print(np.argmax(cnn5_output_p))
# print(cnn5_output_p)
# print("ssim:", diff_ssim(train_images[randIndex], generate_p))

# print("----------------------------------------------------------------------")
# print("modify_output_u1:")
# print(np.argmax(modify_output_u1))
# print(modify_output_u1)
# print("cnn5_output_u1:")
# print(np.argmax(cnn5_output_u1))
# print(cnn5_output_u1)
# print("ssim:", diff_ssim(train_images[randIndex], generate_u1))

print("----------------------------------------------------------------------")
print("modify_output_u2:")
print(np.argmax(modify_output_u2))
print(modify_output_u2)
print("cnn5_output_u2:")
print(np.argmax(cnn5_output_u2))
print(cnn5_output_u2)
print("ssim:", diff_ssim(train_images[randIndex], generate_u2))

# print("----------------------------------------------------------------------")
# print("modify_output_g1:")
# print(np.argmax(modify_output_g1))
# print(modify_output_g1)
# print("cnn5_output_g1:")
# print(np.argmax(cnn5_output_g1))
# print(cnn5_output_g1)
# print("ssim:", diff_ssim(train_images[randIndex], generate_g1))

# print("----------------------------------------------------------------------")
# print("modify_output_g2:")
# print(np.argmax(modify_output_g2))
# print(modify_output_g2)
# print("cnn5_output_g2:")
# print(np.argmax(cnn5_output_g2))
# print(cnn5_output_g2)
# print("ssim:", diff_ssim(train_images[randIndex], generate_g2))

plt.subplot(2, 3, 1)
plt.imshow((train_images[randIndex] * 255).reshape([28, 28]))
# plt.subplot(2, 3, 2)
# plt.imshow((generate_p * 255).reshape([28, 28]))
# plt.subplot(2, 3, 3)
# plt.imshow((generate_u1 * 255).reshape([28, 28]))
plt.subplot(2, 3, 4)
plt.imshow((generate_u2 * 255).reshape([28, 28]))
# plt.subplot(2, 3, 5)
# plt.imshow((generate_g1 * 255).reshape([28, 28]))
# plt.subplot(2, 3, 6)
# plt.imshow((generate_g2 * 255).reshape([28, 28]))
plt.show()
