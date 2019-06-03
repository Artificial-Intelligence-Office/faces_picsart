from itertools import islice
from os.path import join

import numpy as np
from keras_preprocessing.image import load_img


def split_every(batch_size, file_names, dir_images, dir_masks):
    while True:
        file_names = np.random.permutation(file_names)
        i = iter(file_names)
        batch_train_file_names = list(islice(i, batch_size))
        while batch_train_file_names:
            #print(batch_train_file_names)
            x_train_batch = np.array([np.array(load_img(join(dir_images, '{}.jpg'.format(f)), grayscale=False))
                                      / 255 for f in batch_train_file_names])
            y_train_batch = np.array([np.array(load_img(join(dir_masks, '{}.png'.format(f)), grayscale=True))
                                      / 255 for f in batch_train_file_names])
            y_train_batch = y_train_batch.reshape((y_train_batch.shape + (1, )))
            yield x_train_batch, y_train_batch
            batch_train_file_names = list(islice(i, batch_size))
