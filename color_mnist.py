from torchvision.datasets import MNIST
import numpy as np
import random
from PIL import Image
import torch
import cv2


mnist_dataset_train = MNIST('data/mnist/', train=True, download=True)
mnist_dataset_test = MNIST('data/mnist/', train=False, download=True)


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
        # arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
        #                       np.zeros((h, w, 1), dtype=dtype),
        #                       arr], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr



train_set = []

for idx, (im, label) in enumerate(mnist_dataset_train):
    if idx % 1000 == 0:
        print(f'Converting image {idx}/{len(mnist_dataset_train)}')

    im_array = np.array(im)

    color_red = random.choice([True, False])

    colored_arr = color_grayscale_arr(im_array, red=color_red)

    # cv2.imwrite('image_{}.png'.format(idx), colored_arr)

    train_set.append((Image.fromarray(colored_arr), label, color_red))

torch.save(train_set, 'data/cmnist/train.pt')





test_set = []
for idx, (im, label) in enumerate(mnist_dataset_test):
    if idx % 1000 == 0:
        print(f'Converting image {idx}/{len(mnist_dataset_test)}')

    im_array = np.array(im)

    color_red = random.choice([True, False])

    colored_arr = color_grayscale_arr(im_array, red=color_red)

    # cv2.imwrite('image_{}.png'.format(idx), colored_arr)

    test_set.append((Image.fromarray(colored_arr), label, color_red))

torch.save(test_set, 'data/cmnist/test.pt')
