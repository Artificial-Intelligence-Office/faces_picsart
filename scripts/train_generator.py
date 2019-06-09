import os, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать

from scripts.loggger import TensorBoardBatchLogger
from scripts.metrics import dice_coef_K
from scripts.models import build_model
from scripts.utils import split_every

# Constants:

LOAD_MODEL = False
batch_size = 8
epochs = 200
img_size_target = (320, 240)
train_dir = '../data/train/'
masks_dir = '../data/train_mask/'

# Data:
file_names = np.array([f[:f.find('.')] for f in listdir(train_dir) if isfile(join(train_dir, f))])
train_file_names = np.random.permutation(file_names)[:round(0.8 * len(file_names))]
valid_file_names = np.random.permutation(file_names)[round(0.8 * len(file_names)):]

print('Number of files:', len(file_names))
print('Number of train objects:', len(train_file_names))
print('Number of val objects:', len(valid_file_names))

# Model:
if LOAD_MODEL:
    print('Loading model...')
    model = load_model("../models/unet_weights.20-val_loss0.14-0.94.hdf5.model", custom_objects={'dice_coef_K': dice_coef_K})
else:
    # Если обучать с нуля:
    input_layer = Input(img_size_target + (3,))
    output_layer = build_model(input_layer, 16, 0.5)
    model = Model(input_layer, output_layer)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[dice_coef_K])
    model.summary()

# Checkpoints:
model_checkpoint = ModelCheckpoint("../models/unet_weights.{epoch:02d}-val_loss{val_loss:.2f}-{val_dice_coef_K:.2f}.hdf5.model", monitor='val_dice_coef_K',
                                   save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
TB = TensorBoardBatchLogger(project_path=PROJECT_PATH, batch_size=batch_size)

# Training:
history = model.fit_generator(
                    split_every(batch_size, train_file_names, train_dir, masks_dir),
                    steps_per_epoch=len(train_file_names) // batch_size,
                    validation_data=split_every(batch_size, valid_file_names, train_dir, masks_dir), # valid картинки берём просто из train папки
                    validation_steps=len(valid_file_names) // batch_size,
                    epochs=epochs,
                    callbacks=[model_checkpoint, reduce_lr, TB],
                    verbose=1)
print('Fitted!')
