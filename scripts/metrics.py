import keras.backend as K
import numpy as np
import tensorflow as tf

def dice_coef_K(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred, smooth=1):
    intersection = (y_true.flatten() * y_pred.flatten()).sum()
    return -(2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def dice_coef_batch(y_true_in, y_pred_in):
    y_pred_in = (y_pred_in > 0.5).astype(np.float32)  # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = dice_coef_np(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def my_dice_metric(label, pred):
    metric_value = tf.py_func(dice_coef_batch, [label, pred], tf.float64)
    return metric_value
