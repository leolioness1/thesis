# !/usr/bin/env python
# coding: utf-8
import io
from datetime import datetime
from glob import glob
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
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

# In[3]:


# Check if the satellite layers are blank
# invalid = []
# for tiff in tiffs:
#     with rasterio.open(tiff, 'r') as src:
#         arr = src.read()

#         broken_bands = False
#         for ar in arr:

#             if np.array_equal(np.unique(ar), [ 0,  1,  2, 11]):
#                 broken_bands = True
#             if np.array_equal(np.unique(ar), [ 0,  1,  2,  3, 10, 11]):
#                 broken_bands = True

#     if broken_bands:
#         invalid.append(tiff)

# print(invalid)

# for f in invalid:
#     os.remove(f)


# In[4]:


# # Clip rasters to the sites, because for some reason we added multiple sites per geotiff
# clipped = []
# for tiff in tiffs:
#     with rasterio.open(tiff, 'r') as src:
#         arr = src.read()
#         crs = src.crs
#         transform = src.transform

#     sites = h.polygonise(arr[0], threshold=0.5, crs=crs, transform=transform)
#     sites = sites.geometry.to_list()

#     for i, site in enumerate(sites):
#         coords = h.polygon_to_raster_coords(site, crs)

#         name = os.path.splitext(os.path.basename(tiff))[0]
#         name = name + f'_site_{i}'

#         clipped.append(
#             h.clip_rasters([tiff], coords, './data/clipped', [name])[0]
#         )

# len(clipped)


# In[5]:


# Remove all files which don't have polygons in the poly layer
# missing_polys = []
# for tiff in tiffs:
#     with rasterio.open(tiff, 'r') as src:
#         arr = src.read()
#         polys = arr[0]

#     if polys.max() == 0:
#         missing_polys.append(tiff)

# print(missing_polys)

# for f in missing_polys:
#     os.remove(f)


# ### Check Data

# In[3]:


final = sorted(glob.glob('./adam_cnn/cnn/data/cloudless/*.tiff'))

# for f in final:
#     with rasterio.open(f, 'r') as src:
#         print(f)
#         for arr in src.read():
#             plt.imshow(arr)
#             plt.show()

#             w = arr.shape[0]
#             h = arr.shape[1]

#             plt.imshow(arr[w//2:(w//2)+64,h//2:(h//2)+64])
#             plt.show()

#             print(np.unique(arr))
#             print(arr.max())
#         print("==============================")

#
# # In[4]:
#
#
# # Finding the optimal window size to reduce data loss
# # window sizes of 64, 128, 256, 512
# # or 50, 100, 150, 200, 250...
# window_sizes = [64, 128, 256, 512]
#
# results = {}
# for window in window_sizes:
#     results[window] = []
#
# for img in final:
#     with rasterio.open(img, 'r') as src:
#         arr = src.read(1)
#         h = arr.shape[0]
#         w = arr.shape[1]
#
#     for window in window_sizes:
#         h_n = h // window
#         w_n = w // window
#
#         h_loss = h - (window * h_n)
#         w_loss = w - (window * w_n)
#
#         total_loss = (h_loss * h) + (w_loss * w) - (h_loss * w_loss)
#
#         results[window].append([total_loss, total_loss / (h * w)])
#
# for key, value in results.items():
#     results[key] = np.array(value)
#
# # In[5]:
#
#
# fig, axes = plt.subplots(1, 1, figsize=(8, 8))
#
# mean_pct_loss = [x[:, 1].mean() for x in results.values()]
#
# axes.plot(window_sizes, mean_pct_loss)
# axes.set_xlabel('Window Size')
# axes.set_ylabel('% Pixel Loss')
# axes.set_title('Percentage Pixel Loss against Window Size')
# plt.show()
#
# best = np.array(mean_pct_loss).argsort()
# best = np.array(window_sizes)[best[0]]
# print('Best window size: ', best)
# print(list(results.values())[0][:, 1].mean())

# In[6]:


# Create training arrays
window = 64
data = []
for img in final:
    with rasterio.open(img, 'r') as src:
        data.extend(helper.create_training_arrays(src, window))

# In[7]:


bad = []
for i, d in enumerate(data):
    if not d.shape == (6, window, window):
        bad.append(i)

print('Uncleaned: ', len(data))

bad.reverse()

for idx in bad:
    data.pop(idx)

print('Cleaned: ', len(data))

# In[8]:


# Separate positive and negative training images.
# Because we are doing segmentation there is no point having fully negative images, so we will only focus on the pos images
neg = []
pos = []

for d in data:
    pos_sum = (d[0] == 1).sum()
    if pos_sum == 0:
        neg.append(d)
    else:
        pos.append(d)

print('Negative: ', len(neg))
print('Positive: ', len(pos))

# In[9]:


idx = []
for i, p in enumerate(pos):
    layer_max = np.array([x.max() for x in p[2:6]]).max()
    if layer_max == 0:
        idx.append(i)

print(idx)

# # Data Prep


# Seed random
seed = 1004493

# generate train, val, test
train, val_test = train_test_split(pos, test_size=0.2, shuffle=True, random_state=seed)
val, test = train_test_split(val_test, test_size=0.15, shuffle=True, random_state=seed)

print('train: ', len(train))
print('val: ', len(val))
print('test: ', len(test))


datasets = [train, val, test]
names = ['train', 'val', 'test']

data = {}
for ds, name in zip(datasets, names):
    l = []
    for arr in ds:
        l.append(arr[0].sum())
    data[name] = l

plt.hist(sorted(data['train']), bins=20)
plt.title('Train data positive label counts')
plt.show()

plt.hist(sorted(data['val']), bins=20)
plt.title('Val data positive label counts')
plt.show()

plt.hist(sorted(data['test']), bins=20)
plt.title('Test data positive label counts')
plt.show()



# Convert to numpy arrays of shape (len(ds), 8, window, window)
def invert_shape(img):
    trans = np.reshape(np.ravel(img, order='F'), (-1, img.shape[0]), order='C')
    trans = np.reshape(trans, (img.shape[2], img.shape[1], img.shape[0]), order='F')
    return trans


# In[13]:


np_train = np.empty((len(train), window, window, 6))

for i, data in enumerate(train):
    np_train[i] = invert_shape(data)
print(np_train.shape)

np_val = np.empty((len(val), window, window, 6))
for i, data in enumerate(val):
    np_val[i] = invert_shape(data)
print(np_val.shape)

np_test = np.empty((len(test), window, window, 6))
for i, data in enumerate(test):
    np_test[i] = invert_shape(data)

print(np_test.shape)

# In[14]:


# Split into X and Y components
train_X = np_train[:, :, :, 2:]
train_Y = np_train[:, :, :, 0]
train_Y = np.reshape(train_Y, list(train_Y.shape) + [1])

val_X = np_val[:, :, :, 2:]
val_Y = np_val[:, :, :, 0]
val_Y = np.reshape(val_Y, list(val_Y.shape) + [1])

test_X = np_test[:, :, :, 2:]
test_Y = np_test[:, :, :, 0]
test_Y = np.reshape(test_Y, list(test_Y.shape) + [1])

print(test_X.shape)
print(test_Y.shape)

# # Preprocessing

# In[16]:


# # Fit scaler to train data, defualt is MinMaxScaler()
# scaler = Scaler(train_X)
# scaler.fit_scaler()
#
# # In[19]:
#
#
# # Scale all datasets
# scaled_train_X = np.empty(train_X.shape)
# for i, arr in enumerate(train_X):
#     scaled_train_X[i] = train_X[i] / 10000
#
# scaled_val_X = np.empty(val_X.shape)
# for i, arr in enumerate(val_X):
#     scaled_val_X[i] = val_X[i] / 10000
#
# scaled_test_X = np.empty(test_X.shape)
# for i, arr in enumerate(test_X):
#     scaled_test_X[i] = test_X[i] / 10000

# scaler_b1 = Scaler(train_X[i], scaler= StandardScaler())
# scaler.fit_scaler()
#
# # In[19]:
# for i, arr in enumerate(train_X):
#     scaler = Scaler(train_X[i], scaler=StandardScaler())
#     scaler.fit_scaler()



# Scale all datasets
scaled_train_X = np.empty(train_X.shape)
for i, arr in enumerate(train_X):
    scaler = Scaler(train_X[i], scaler=StandardScaler())
    scaler.fit_scaler()
    scaled_train_X[i] = scaler.transform(arr)

scaled_val_X = np.empty(val_X.shape)
for i, arr in enumerate(val_X):
    scaler = Scaler(train_X[i], scaler=StandardScaler())
    scaler.fit_scaler()
    scaled_val_X[i] = scaler.transform(arr)

scaled_test_X = np.empty(test_X.shape)
for i, arr in enumerate(test_X):
    scaler = Scaler(train_X[i], scaler=StandardScaler())
    scaler.fit_scaler()
    scaled_test_X[i] = scaler.transform(arr)

# In[20]:


shape = (window, window, 4)

# Add code for resize
# Add code for normalize range to 0-1
# Add code fro augmentations

# x_ex=train_generator.__getitem__(1)
#
# import mlflow
#
# experiment_name = 'Baseline Segmentation: elu, he_normal init, nadam'
# mlflow.set_experiment(experiment_name)
# mlflow.tensorflow.autolog()


smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    tf.summary.scalar('jaccard_coef', data=K.mean(jac))
    return K.mean(jac)


def iou_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)


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


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    tf.summary.scalar('jaccard_coef_int', data=K.mean(jac))
    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


"""
Define our custom loss function.
"""


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


norm_method = 'z_score'
test_generator_gt = ImageGenerator(scaled_test_X, test_Y, dim=(shape[0], shape[1]),
                                                         n_channels=shape[2],batch_size=len(test_Y))
test_gt = test_generator_gt.__getitem__(0)
test_gt[0][-1].shape
test_gt[1][0].shape
x_example_ex = np.expand_dims(test_gt[0][0], axis=0)
mask_example_ex = np.expand_dims(test_gt[1][0], axis=0)

print('Running predictions...')
# model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files\model_3_crossentropy_dice_loss_run-2_50'
# model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files\model_z_score_3_crossentropy_dice_loss_adam_run-36_100_aug'
# model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_first_selection\model_z_score_4_crossentropy_dice_loss_adam_0.0001_64_20'
# model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_optimiser_batch_lr_selection\model_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_50' #best so far
# model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_loss_he_naive_selection\model_naive_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_normal_100'
# model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_opt_lr_selection\model_z_score_4_crossentropy_dice_loss_nadam_0.0001_64_elu_he_uniform_200'
#model_path =r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_recontruct_2\model_z_score_4_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_uniform_200'
# model_path =r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_z_score_selection\model_z_score_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_uniform_200'
# model_path =r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_z_score_batch_selection\model_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_gelu_he_normal_200'
# model_path=r'model_files_adam_new_dataloss_selection\model_z_score_10_dice_coef_loss_rmsprop_0.001_64_gelu_he_normal_100'
model_path =r'model_files_adam_new_data_recontruct_3\model_z_score_4_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_uniform_200' #0.76 test dice coeff

reconstructed_model = tf.keras.models.load_model(model_path, compile=False)
reconstructed_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001,rho=0.9, epsilon=None,decay=0.0),
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