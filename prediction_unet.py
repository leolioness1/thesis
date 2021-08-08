# coding: utf-8
from glob import glob
import rasterio
import rasterio.mask
import geopandas as gpd
import fiona
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
from tensorflow.keras import backend as K
import random
from data_generator_segmentation import DataGenerator_segmentation

TEST_LOCAL_PATH =  '../Perma_Thesis/test_data'

#
#
# random.seed(123)
# np.random.seed(123)
# tf.random.set_seed(123)
#



# def load_image(image_path):
#       """Load grayscale image
#       :param image_path: path to image to load
#       :return: loaded image
#       """
#       img_object = rasterio.open(image_path)
#       img=img_object.read()
#       #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
#       channels=3
#       size=64
#       img_temp = img[:channels,:256,:256]
#       img_temp[img_temp > 1000] = 1000
#       img_temp = img_temp / 1000
#       img_final = np.moveaxis(img_temp, 0, -1)
# #         #Reducing image size to 40*40 crop from centre based on finding on 12/03 on thaw slump size being avg
# #         #400m so 40 pixels
# #         startx = 98 #(128-size/2)
# #         starty = 98 #(128-size/2)
# #         img_final = img_final[startx:startx+size,startx:starty+size,:]
#       return img_final
#
def load_mask(image_path):
    img_object = rasterio.open(image_path)
    img = img_object.read()
    # Selecting only 3 channels and fixing size to 256 not correct way exactly but hack

    mask = img[-1, :256, :256]
    mask_final = np.moveaxis(mask, 0, -1)
    np.nan_to_num(mask_final, nan=0, copy=False)  # Change nans from data to 0 for mask
    return mask_final


from data_generator_segmentation import DataGenerator_segmentation

# image_test=load_image(image_path='../Perma_Thesis/MSI/thaw/25-20190905_195023_1032.tif')
#
# mask_test= load_mask(image_path='../Perma_Thesis/MSI/thaw/25-20190905_195023_1032.tif')
#
# print( "Test Image dimensions:" + str(image_test.shape))
#
# print( "Test Mask dimensions:" + str(mask_test.shape))
#
# plt.imshow(image_test)
# plt.show()
#
# plt.imshow(mask_test)
# plt.show()


# # Proves that crop to size 40*40 is better but maybe make it 100*100??
# img_cropped = image_test[44:84, 44:84,:]
# plt.imshow(img_cropped)
# plt.show()
# #but still images are very bad quality resolution and need to add more bands!!!!
#
# validated_shape_files = gpd.read_file(VALIDATED_SHAPE_FILE_PATH)
# validated_shape_files = validated_shape_files.to_crs("EPSG:32604")
# shapes = validated_shape_files['geometry']

#
# IMG_HEIGHT=256
# IMG_WIDTH=256
# BATCH_SIZE = 8


# ##### Load image, preprocess, augment through image data generator

# # #### Custom data generator
#
# df_dataset = pd.DataFrame()
# files = glob(LOCAL_PATH + "**/*.tif")
#
#
# df_dataset['image_paths'] = files
# df_dataset['labels_string'] = df_dataset['image_paths'].str.split('\\').str[-2]
# df_dataset['label'] = df_dataset['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
# df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)  # Randomize
# df_dataset['image_paths'].str.split('\\').str[-2]
# df_dataset

df_test = pd.DataFrame()
test_files = glob(TEST_LOCAL_PATH + "**/*.tif")
df_test['image_paths'] = test_files
df_test['labels_string'] = df_test['image_paths'].str.split('\\').str[-2]
df_test['label'] =  df_test['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_test=df_test.sample(frac=1).reset_index(drop=True) #Randomize
df_test

# print("Full Dataset label distribution")
# print(df_dataset.groupby('labels_string').count())
#
# train, val = train_test_split(df_dataset, test_size=0.149, random_state=123)
# #val, test = train_test_split(val, test_size=0.25, random_state=123)
# print("\nTrain Dataset label distribution")
# print(train.groupby('labels_string').count())
# print("\nVal Dataset label distribution")
# print(val.groupby('labels_string').count())
print("\nTest Dataset label distribution")
print(df_test.groupby('labels_string').count())

# Custom data generator that replaces PIL function on image read of tiff record with rasterio
# as default Imagedatagenertator seems to be throwing error

from data_generator_segmentation import DataGenerator_segmentation

#
def load_mask(image_path):
    img_object = rasterio.open(image_path)
    img = img_object.read()
    # Selecting only 3 channels and fixing size to 256 not correct way exactly but hack

    mask = img[-1, :256, :256]
    mask_final = np.moveaxis(mask, 0, -1)
    np.nan_to_num(mask_final, nan=0, copy=False)  # Change nans from data to 0 for mask
    return mask_final


# from data_generator_segmentation import DataGenerator_segmentation
#
# # image_test=load_image(image_path='../Perma_Thesis/MSI/thaw/25-20190905_195023_1032.tif')
# #
# # mask_test= load_mask(image_path='../Perma_Thesis/MSI/thaw/25-20190905_195023_1032.tif')
# #
# # print( "Test Image dimensions:" + str(image_test.shape))
# #
# # print( "Test Mask dimensions:" + str(mask_test.shape))
# #
# # plt.imshow(image_test)
# # plt.show()
# #
# # plt.imshow(mask_test)
# # plt.show()
#
#
# # # Proves that crop to size 40*40 is better but maybe make it 100*100??
# # img_cropped = image_test[44:84, 44:84,:]
# # plt.imshow(img_cropped)
# # plt.show()
# # #but still images are very bad quality resolution and need to add more bands!!!!
# #
# # validated_shape_files = gpd.read_file(VALIDATED_SHAPE_FILE_PATH)
# # validated_shape_files = validated_shape_files.to_crs("EPSG:32604")
# # shapes = validated_shape_files['geometry']
#
# #
# # IMG_HEIGHT=256
# # IMG_WIDTH=256
# # BATCH_SIZE = 8
#
#
# # ##### Load image, preprocess, augment through image data generator
#
# # #### Custom data generator
#
# df_dataset = pd.DataFrame()
# files = glob(LOCAL_PATH + "**/*.tif")
# df_dataset['image_paths'] = files
# df_dataset['labels_string'] = df_dataset['image_paths'].str.split('\\').str[-2]
# df_dataset['label'] = df_dataset['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
# df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)  # Randomize
# df_dataset['image_paths'].str.split('\\').str[-2]
# df_dataset
#
# # df_val = pd.DataFrame()
# # files_val = glob(VAL_LOCAL_PATH + "**/*.tif")
# # df_val['image_paths'] = files_val
# # df_val['labels_string'] = df_val['image_paths'].str.split('\\').str[-2]
# # df_val['label'] =  df_val['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
# # df_val=df_val.sample(frac=1).reset_index(drop=True) #Randomize
# # df_val
#
# print("Full Dataset label distribution")
# print(df_dataset.groupby('labels_string').count())
#
# train, val = train_test_split(df_dataset, test_size=0.2, random_state=123)
# val, test = train_test_split(val, test_size=0.25, random_state=123)
# print("\nTrain Dataset label distribution")
# print(train.groupby('labels_string').count())
# print("\nVal Dataset label distribution")
# print(val.groupby('labels_string').count())
# print("\nTest Dataset label distribution")
# print(test.groupby('labels_string').count())


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


def dice_coeff(y_true, y_pred, smooth=1):
    """
    Arguments:
        y_true: (string) ground truth image mask
        y_pred : (int) predicted image mask

    Returns:
        Calculated Dice coeffecient
    """
    y_true_f = K.flatten(y_true)
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    #tf.summary.scalar('dice_coef', data=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    tf.summary.scalar('dice_coef', data=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def crossentropy_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


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

# Custom data generator that replaces PIL function on image read of tiff record with rasterio
# as default Imagedatagenertator seems to be throwing error

from data_generator_segmentation import DataGenerator_segmentation



norm_method = 'z_score'
test_generator_gt = DataGenerator_segmentation(df_test, dimension=(64, 64), size=64, norm_method=norm_method,
                                               batch_size=len(df_test), n_channels=4)
test_gt = test_generator_gt.__getitem__(0)
test_gt[0][-1].shape
test_gt[1][0].shape
x_example_ex = np.expand_dims(test_gt[0][0], axis=0)
mask_example_ex = np.expand_dims(test_gt[1][0], axis=0)
print('Running predictions...')
model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files\model_3_crossentropy_dice_loss_run-2_50'
model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files\model_z_score_3_crossentropy_dice_loss_adam_run-36_100_aug'
model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_first_selection\model_z_score_4_crossentropy_dice_loss_adam_0.0001_64_20'
model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_optimiser_batch_lr_selection\model_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_50' #best so far
model_path = r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_z_score_loss_selection\model_z_score_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_uniform_100'

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


