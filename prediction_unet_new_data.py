# !/usr/bin/env python
# coding: utf-8
import io
from datetime import datetime
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, UpSampling2D, \
    BatchNormalization, Activation, Add, Multiply, Dropout, Lambda, MaxPooling2D, concatenate, Dense, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import random
from tensorboard.plugins.hparams import api as hp
import shutil

# seed_value= 0
# random.seed(seed_value)
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)
# os.environ['PYTHONHASHSEED']=str(seed_value)
#
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.get_session(sess)
# using just positive image labels
# !/usr/bin/env python
# coding: utf-8
import sys
import os

sys.path.append('./adam_cnn/cnn/scripts')

# In[2]:
import rasterio
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helper
from classes import Scaler, ImageGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# tiffs = sorted(glob.glob('./adam_cnn/cnn/data/cloudless/*.tiff'))


# # Data Cleansing
#
#     - Checking which tiles have missing satellite layers and remove
#     - Clipping tiles to sites
#     - Removing sites with no polygons
shape = (64, 64, 4)

# Add code for resize
# Add code for normalize range to 0-1
# Add code fro augmentations

smooth = 1e-12


# def jaccard_coef(y_true, y_pred):
#     intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
#     sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
#
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     tf.summary.scalar('jaccard_coef', data=K.mean(jac))
#     return K.mean(jac)
#worked for iou loss
def jaccard_coef_int(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))
    jac = (intersection + 1e-12) / (sum_ - intersection + 1e-12)
    tf.summary.scalar('jaccard_coef_int', data=jac)
    return jac



def iou_loss(y_true, y_pred):
    return 1 - jaccard_coef_int(y_true, y_pred)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef_int(y_true, y_pred)) + binary_crossentropy(y_true, y_pred)

# def dice_coef(y_true, y_pred, smooth=1):
#     """
#     Arguments:
#         y_true: (string) ground truth image mask
#         y_pred : (int) predicted image mask
#
#     Returns:
#         Calculated Dice coeffecient
#     """
#     y_true_f = K.flatten(y_true)
#     y_true_f = K.cast(y_true_f, 'float32')
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Arguments:
        y_true: (string) ground truth image mask
        y_pred : (int) predicted image mask

    Returns:
        Calculated Dice coeffecient loss
    """
    return 1 - dice_coef(y_true, y_pred)


# def jaccard_coef_int(y_true, y_pred):
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
#     sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     tf.summary.scalar('jaccard_coef_int', data=K.mean(jac))
#     return K.mean(jac)



# def dice_coeff(y_true, y_pred, smooth=1):
#     """
#     Arguments:
#         y_true: (string) ground truth image mask
#         y_pred : (int) predicted image mask
#
#     Returns:
#         Calculated Dice coeffecient
#     """
#     y_true_f = K.flatten(y_true)
#     y_true_f = K.cast(y_true_f, 'float32')
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     #tf.summary.scalar('dice_coef', data=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# def dice_coef(y_true, y_pred, smooth=1e-12):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     intersection = K.sum(K.abs(y_true * y_pred))
#     tf.summary.scalar('dice_coef', data=(2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth))
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_coef(y_true, y_pred, smooth=1e-12):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true = K.cast(y_true, 'float32')
    intersection = K.sum(K.abs(y_true * y_pred))
    tf.summary.scalar('dice_coef', data=(2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def crossentropy_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


# def dice_loss(y_true, y_pred):
#     return 1 - dice_coeff(y_true, y_pred)
#
#
# def crossentropy_coeff_dice_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


import pickle
i='new'
with open(f'train_X{i}.pkl','rb') as f:  train_X = pickle.load(f)
with open(f'val_X{i}.pkl','rb') as f: val_X = pickle.load(f)
with open(f'test_X{i}.pkl','rb') as f: test_X = pickle.load(f)
with open(f'train_Y{i}.pkl','rb') as f: train_Y = pickle.load(f)
with open(f'val_Y{i}.pkl','rb') as f: val_Y = pickle.load(f)
with open(f'test_Y{i}.pkl','rb') as f: test_Y = pickle.load(f)

norm_method = 'naive'

# # Preprocessing

if norm_method == 'naive':
    # Scale all datasets
    print(f'Using {norm_method} normalisation')
    scaled_train_X = train_X / 10000

    scaled_val_X = val_X / 10000

    scaled_test_X = test_X / 10000

elif norm_method == 'z_score':
    print(f'Using {norm_method} normalisation')
    # Scale all datasets per band
    scaled_train_X = np.empty(train_X.shape)
    scaled_val_X = np.empty(val_X.shape)
    scaled_test_X = np.empty(test_X.shape)

    for i, arr in enumerate(train_X):
        scaler = Scaler(train_X[i], scaler=StandardScaler())
        scaler.fit_scaler()
        scaled_train_X[i] = scaler.transform(arr)


    for i, arr in enumerate(val_X):
        scaler = Scaler(train_X[i], scaler=StandardScaler())
        scaler.fit_scaler()
        scaled_val_X[i] = scaler.transform(arr)


    for i, arr in enumerate(test_X):
        scaler = Scaler(train_X[i], scaler=StandardScaler())
        scaler.fit_scaler()
        scaled_test_X[i] = scaler.transform(arr)
else:
    print(f'Using {norm_method} normalisation')
    # Scale all datasets per band
    scaled_train_X = np.empty(train_X.shape)
    scaled_val_X = np.empty(val_X.shape)
    scaled_test_X = np.empty(test_X.shape)

    for i, arr in enumerate(train_X):
        scaler = Scaler(train_X[i], scaler=MinMaxScaler())
        scaler.fit_scaler()
        scaled_train_X[i] = scaler.transform(arr)

    for i, arr in enumerate(val_X):
        scaler = Scaler(train_X[i], scaler=MinMaxScaler())
        scaler.fit_scaler()
        scaled_val_X[i] = scaler.transform(arr)

    for i, arr in enumerate(test_X):
        scaler = Scaler(train_X[i], scaler=MinMaxScaler())
        scaler.fit_scaler()
        scaled_test_X[i] = scaler.transform(arr)

test_generator_gt = ImageGenerator(scaled_test_X, test_Y, dim=(shape[0], shape[1]),
                                                         n_channels=shape[2],batch_size=len(test_Y))
test_gt = test_generator_gt.__getitem__(0)

x_example_ex = np.expand_dims(test_gt[0][0], axis=0)
mask_example_ex = np.expand_dims(test_gt[1][0], axis=0)

print(f'Running predictions for {len(test_Y)} images...')
# model_path='model_files_adam_new_data_recontruct_naive_balanced/model_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200from_scratch'
#model_path='model_files_adam_new_data_naive_planetary\model_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200'
model_path='model_files_final_experiment_more_data/model_naive_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_normal_100' #new data ''
model_path='model_files_final_experiment_more_data_planetary/model_naive_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_normal_100' #planetary data 'new'

reconstructed_model = tf.keras.models.load_model(model_path, compile=False)
reconstructed_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001,rho=0.9, epsilon=None,decay=0.0),
                            loss=crossentropy_dice_loss,
                            metrics=['accuracy',
                                     jaccard_coef_int,
                                     dice_coef,
                                     "binary_crossentropy"
                                     ])
predictions = reconstructed_model.predict(test_gt[0], verbose=1)
score = reconstructed_model.evaluate(test_gt[0], test_gt[1], verbose=1)


def plot_sample(ix=None):
    """Function to plot the results"""

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))  # 4
    ax[0].imshow(test_gt[0][ix])

    ax[0].contour(test_gt[1][ix].squeeze(), colors='r', levels=[0.5])
    ax[0].set_title('Image')

    ax[1].imshow(test_gt[1][ix].squeeze())
    ax[1].set_title('Mask')

    ax[2].imshow(predictions[ix].squeeze(), vmin=0, vmax=1)

    ax[2].contour(test_gt[1][ix].squeeze(), colors='r', levels=[0.5])
    ax[2].set_title('Predicted')
    plt.show()

list = []
for i in range(0, len(predictions)):
    list.append(np.max(predictions[i]))
    plot_sample(i)

print(list)
print(score)