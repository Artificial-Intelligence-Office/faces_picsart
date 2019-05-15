from itertools import islice
from os import listdir
from os.path import isfile, join
import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_preprocessing.image import load_img
from model import get_unet

def split_every(batch_size, iterable):
    i = iter(iterable)
    batch_train_file_names = list(islice(i, batch_size))
    while batch_train_file_names:
        #print(batch_train_file_names)
        x_train_batch = np.array([np.array(load_img(join(train_dir, '{}.jpg'.format(f)), grayscale=False))
                                  / 255 for f in batch_train_file_names])
        y_train_batch = np.array([np.array(load_img(join(train_masks_dir, '{}.png'.format(f)), grayscale=True))
                                  / 255 for f in batch_train_file_names]).reshape(((batch_size,) + img_size_target + (1, )))
        yield x_train_batch, y_train_batch
        batch_train_file_names = list(islice(i, batch_size))

train_dir = 'train'
train_masks_dir = 'train_mask'
file_names = np.array([f[:f.find('.')] for f in listdir(train_dir) if isfile(join(train_dir, f))])

train_file_names = np.random.permutation(file_names)[:round(0.8 * len(file_names))]
valid_file_names = np.random.permutation(file_names)[round(0.8 * len(file_names)):]



print(len(file_names))
print(len(set(train_file_names)))
print(len(train_file_names))
print(len(valid_file_names))
print(len(train_file_names) + len(valid_file_names))


img_size_target = (320, 240)

# Если обучать с нуля:
input_img = Input(img_size_target + (3,), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.summary()


model_checkpoint = ModelCheckpoint("models/unet_weights.{epoch:02d}-val_loss{val_loss:.2f}-{val_dice_coef_K:.2f}.hdf5.model",
                                   monitor='val_dice_coef_K', mode='min', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3, min_lr=0.00001, verbose=1)


batch_size = 16

history = model.fit_generator(
                    split_every(batch_size, train_file_names),
                    steps_per_epoch=len(train_file_names) // batch_size,
                    epochs=3,
                    callbacks=[model_checkpoint, reduce_lr],
                    verbose=1)
print('Fitted!')