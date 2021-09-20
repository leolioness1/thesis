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
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.get_session(sess)
# using just positive image labels
# !/usr/bin/env python
# coding: utf-8
import sys
import os
import rasterio
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helper
from classes import Scaler, ImageGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
sys.path.append('./adam_cnn/cnn/scripts')


#final = sorted(glob.glob('./adam_cnn/cnn/data/cloudless/*.tiff'))

# final = sorted(glob.glob('./validated/*.tiff'))
# final_test = sorted(glob.glob('./validated_test/*.tiff'))
# Create training arrays
window = 64
# data = []
# for img in final:
#     with rasterio.open(img, 'r') as src:
#         data.extend(helper.create_training_arrays(src, window))
#
# test = []
# for img in final_test:
#     with rasterio.open(img, 'r') as src:
#         test.extend(helper.create_training_arrays(src, window))
#
# print(len(data))
# print(len(test))
#
# bad = []
# for i, d in enumerate(data):
#     if not d.shape == (6,window,window):
#         bad.append(i)
#
# print('Uncleaned: ', len(data))
#
# bad.reverse()
#
# for idx in bad:
#     data.pop(idx)
#
# print('Cleaned: ', len(data))
#
#
# bad = []
# for i, d in enumerate(test):
#     if not d.shape == (6,window,window):
#         bad.append(i)
#
# print('Uncleaned: ', len(test))
#
# bad.reverse()
#
# for idx in bad:
#     test.pop(idx)
#
# print('Cleaned: ', len(test))
# # Separate positive and negative training images.
# # Because we are doing segmentation there is no point having fully negative images, so we will only focus on the pos images
# dneg = []
# dpos = []
#
# for d in data:
#     pos_sum = (d[0] == 1).sum()
#     if pos_sum == 0:
#         dneg.append(d)
#     else:
#         dpos.append(d)
#
# print('Negative: ', len(dneg))
# print('Positive: ', len(dpos))
#
# tneg = []
# tpos = []
#
# for d in test:
#     pos_sum = (d[0] == 1).sum()
#     if pos_sum == 0:
#         tneg.append(d)
#     else:
#         tpos.append(d)
#
# print('Negative: ', len(tneg))
# print('Positive: ', len(tpos))
#
# # Double check that data arrays all have satellite data
# idx = []
# for i, p in enumerate(dpos):
#     layer_max = np.array([x.max() for x in p[2:6]]).max()
#     if layer_max == 0:
#         idx.append(i)
#
# idx.reverse()
# print('Uncleaned: ', len(dpos))
#
# for i in idx:
#     dpos.pop(i)
#
# print('Cleaned: ', len(dpos))
#
# idx = []
# for i, p in enumerate(tpos):
#     layer_max = np.array([x.max() for x in p[2:6]]).max()
#     if layer_max == 0:
#         idx.append(i)
#
# idx.reverse()
# print('Uncleaned: ', len(tpos))
#
# for i in idx:
#     tpos.pop(i)
#
# print('Cleaned: ', len(tpos))
#
#
# # data = []
# # for img in final:
# #     with rasterio.open(img, 'r') as src:
# #         data.extend(helper.create_training_arrays(src, window))
# #
# # # Separate positive and negative training images.
# # # Because we are doing segmentation there is no point having fully negative images, so we will only focus on the pos images
# #
# # neg = []
# # pos = []
# #
# # for d in data:
# #     pos_sum = (d[0] == 1).sum()
# #     if pos_sum == 0:
# #         neg.append(d)
# #     else:
# #         pos.append(d)
# #
# # print('Negative: ', len(neg))
# # print('Positive: ', len(pos))
# #
# # # Create Equal Split of Pos and Neg
# # seed= 42
# # np.random.seed(seed)
# #
# # neg = np.array(neg)
# #
# # np.random.shuffle(neg)
# # neg = list(neg[:len(pos), :, : ,:])
# # dcombined = pos + neg
# #
# # #
# # # tneg = np.array(tneg)
# # #
# # # np.random.shuffle(tneg)
# # # tneg = list(tneg[:len(tpos), :, : ,:])
# # # tcombined = tpos + tneg
# #
# # print(len(dcombined))
# # # print(len(tcombined))
# #
# #
# # # Double check that data arrays all have satellite data
# # idx = []
# # for i, p in enumerate(dcombined):
# #     layer_max = np.array([x.max() for x in p[2:6]]).max()
# #     if layer_max == 0:
# #         idx.append(i)
# #
# # idx.reverse()
# # print('Uncleaned: ', len(dcombined))
# #
# # for i in idx:
# #     pos.pop(i)
# #
# # print('Cleaned: ', len(dcombined))
# #
# #
# # # # Data Prep
# # # Seed random
# seed = 1004493
#
# # generate train, val, test
# # train, val_test = train_test_split(dcombined, test_size=0.2, shuffle=True, random_state=seed)
# # val, test = train_test_split(val_test, test_size=0.23, shuffle=True, random_state=seed)
# train=dpos
# val, test = train_test_split(tpos, test_size=0.25, shuffle=True, random_state=seed)
# #
# print('train: ', len(train))
# print('val: ', len(val))
# print('test: ', len(test))
# #
# #
# datasets = [train, val, test]
# names = ['train', 'val', 'test']
#
# data_plot = {}
# for ds, name in zip(datasets, names):
#     l = []
#     for arr in ds:
#         l.append(arr[0].sum())
#     data_plot[name] = l
#
# plt.hist(sorted(data_plot['train']), bins=20)
# # plt.title('Train data positive label counts')
# # plt.show()
#
# plt.hist(sorted(data_plot['val']), bins=20)
# # plt.title('Val data positive label counts')
# # plt.show()
#
# plt.hist(sorted(data_plot['test']), bins=20)
# plt.title('Data positive label counts')
# plt.legend(['Train','Validation', 'Test'])
# plt.show()
# #
# #
# #
# # Convert to numpy arrays of shape (len(ds), 8, window, window)
# def invert_shape(img):
#     trans = np.reshape(np.ravel(img, order='F'), (-1, img.shape[0]), order='C')
#     trans = np.reshape(trans, (img.shape[2], img.shape[1], img.shape[0]), order='F')
#     return trans
#
# np_train = np.empty((len(train), window, window, 6))
#
# for i, data in enumerate(train):
#     np_train[i] = invert_shape(data)
#
# np_val = np.empty((len(val), window, window, 6))
# for i, data in enumerate(val):
#     np_val[i] = invert_shape(data)
#
# np_test = np.empty((len(test), window, window, 6))
# for i, data in enumerate(test):
#     np_test[i] = invert_shape(data)
#
#
#
# # Split into X and Y components
# train_X = np_train[:, :, :, 2:]
# train_Y = np_train[:, :, :, 0]
# train_Y = np.reshape(train_Y, list(train_Y.shape) + [1])
# print(train_X.shape)
# print(train_Y.shape)
# val_X = np_val[:, :, :, 2:]
# val_Y = np_val[:, :, :, 0]
# val_Y = np.reshape(val_Y, list(val_Y.shape) + [1])
# print(val_X.shape)
# print(val_Y.shape)
#
# test_X = np_test[:, :, :, 2:]
# test_Y = np_test[:, :, :, 0]
# test_Y = np.reshape(test_Y, list(test_Y.shape) + [1])
#
# print(test_X.shape)
# print(test_Y.shape)
i='new' #balanced
# import pickle
# with open(f'train_X{i}.pkl','wb') as f: pickle.dump(train_X, f)
# with open(f'val_X{i}.pkl','wb') as f: pickle.dump(val_X, f)
# with open(f'test_X{i}.pkl','wb') as f: pickle.dump(test_X, f)
# with open(f'train_Y{i}.pkl','wb') as f: pickle.dump(train_Y, f)
# with open(f'val_Y{i}.pkl','wb') as f: pickle.dump(val_Y, f)
# with open(f'test_Y{i}.pkl','wb') as f: pickle.dump(test_Y, f)

import pickle
i='new'
with open(f'train_X{i}.pkl','rb') as f:  train_X = pickle.load(f)
with open(f'val_X{i}.pkl','rb') as f: val_X = pickle.load(f)
with open(f'test_X{i}.pkl','rb') as f: test_X = pickle.load(f)
with open(f'train_Y{i}.pkl','rb') as f: train_Y = pickle.load(f)
with open(f'val_Y{i}.pkl','rb') as f: val_Y = pickle.load(f)
with open(f'test_Y{i}.pkl','rb') as f: test_Y = pickle.load(f)


# Scale all datasets
scaled_train_X = np.empty(train_X.shape)
for i, arr in enumerate(train_X):
    scaled_train_X[i] = train_X[i] / 10000

scaled_val_X = np.empty(val_X.shape)
for i, arr in enumerate(val_X):
    scaled_val_X[i] = val_X[i] / 10000

scaled_test_X = np.empty(test_X.shape)
for i, arr in enumerate(test_X):
    scaled_test_X[i] = test_X[i] / 10000






#Fit scaler to train data, defualt is MinMaxScaler()
# scaler_type=StandardScaler() #MinMaxScaler()#
# # Scale all datasets per band
# scaled_train_X = np.empty(train_X.shape)
# scaled_val_X = np.empty(val_X.shape)
# scaled_test_X = np.empty(test_X.shape)
#
# for i, arr in enumerate(train_X):
#     scaler = Scaler(train_X[i], scaler=scaler_type)
#     scaler.fit_scaler()
#     scaled_train_X[i] = scaler.transform(arr)
#
#
# for i, arr in enumerate(val_X):
#     scaler = Scaler(train_X[i], scaler=scaler_type)
#     scaler.fit_scaler()
#     scaled_val_X[i] = scaler.transform(arr)
#
#
# for i, arr in enumerate(test_X):
#     scaler = Scaler(train_X[i], scaler=scaler_type)
#     scaler.fit_scaler()
#     scaled_test_X[i] = scaler.transform(arr)

shape = (window, window, 4)
smooth = 1e-12

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


from datetime import time

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.clock()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append((epoch, time.clock() - self.timetaken))

    def on_train_end(self, logs={}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()


def get_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=4, activation_func='elu', init_method='he_normal'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
    # t1 = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2,0.2, 'nearest', interpolation = 'bilinear')(inputs)
    # t2 = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, 'nearest', interpolation='bilinear')(t1)
    c1 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(inputs)
    c1 = Dropout(0.1, )(c1)
    c1 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(p1)
    c2 = Dropout(0.1, )(c2)
    c2 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(p2)
    c3 = Dropout(0.2, )(c3)
    c3 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(p3)
    c4 = Dropout(0.2, )(c4)
    c4 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(p4)
    c5 = Dropout(0.3, )(c5)
    c5 = Conv2D(256, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c5)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(u6)
    c6 = Dropout(0.2, )(c6)
    c6 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(u7)
    c7 = Dropout(0.2, )(c7)
    c7 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(u8)
    c8 = Dropout(0.1, )(c8)
    c8 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(u9)
    c9 = Dropout(0.1, )(c9)
    c9 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer=init_method, padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, mean_iou(2)])
    return model


# MODEL_DIR = ''
# m = tf.keras.models.load_model(MODEL_DIR)
# m.summary()

# params = {"height": height
#     , "width": width
#     ,"n_channels": n_channels
#     ,"normalisation": ">10000/10000",
#      "model": "UNET"}

# logdir= 'logs/hparam_tuning'
# logs_dir = "logs/scalars/" + datetime.now().strftime("%Y%m%d%H%M%S")
# dice_list = []
# dice_val_list = []
# jaccard_list = []
# jaccard_val_list = []
# loss_list = []
# batch_list = []
# CHANGEME

experiment_folder = 'adam_new_data_recontruct_naive_1'
for i in ['model_files', 'history_files', 'weights_files', 'plots']:
    if os.path.exists(f'{i}_{experiment_folder}'):
        print('already here')
        pass
    else:
        os.mkdir(f'{i}_{experiment_folder}')


def train_test_model(hparams, run_dir, name, n_epochs=5):
    # 5. Set the callbacks for saving the weights and the tensorboard
    # + "/lr_{}".format(round(learning_rate,8))
    height = hparams[PATCH_SIZE]
    width = hparams[PATCH_SIZE]
    activation_name = hparams[ACTIVATION]
    init_name = hparams[INITIALISATION]
    n_channels = 4
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=run_dir, histogram_freq=0, update_freq="epoch")
    terminate_nan=tf.keras.callbacks.TerminateOnNaN()
    model = get_unet(IMG_WIDTH=height, IMG_HEIGHT=width, IMG_CHANNELS=n_channels, activation_func=activation_name, init_method=init_name)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto',  restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1, mode='min')
    optimiser_name = hparams[HP_OPTIMIZER]
    norm_name = hparams[HP_NORM]
    # optimiser_name = 'adam'
    loss_name = hparams[LOSS]
    batch_size = hparams[BATCH_SIZE]
    # loss_list.append(loss_name)
    # batch_list.append(batch_size)
    learning_rate = hparams[HP_LEARNING_RATE]
    now_time = datetime.now().strftime("%Y%m%d%H%M%S")
    weights_path = fr"C:\Users\leo__\PycharmProjects\Perma_Thesis\weights_files_{experiment_folder}\weights_{name}.hdf5"
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)
    # 4. Select the optimizer and the learning rate (default option is Adam)
    # epsilon=1e-07, A small constant for numerical stability
    if optimiser_name == 'rmsprop':
        optimiser = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=None, decay=0.0)  # default momentum =0.0, 0.9 did not go well xD
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop
        # This implementation of RMSprop uses plain momentum, not Nesterov momentum.
        # The centered version additionally maintains a moving average of the gradients, and uses that average to estimate the variance.
    elif optimiser_name == 'nadam':
        optimiser = tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0)  # )  # learning_rate=learning_rate
    elif optimiser_name == 'sgd':
        optimiser = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=False, decay=0.0)  # default momentum =0.0
    elif optimiser_name == 'adam':
        optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,decay=0.0,
                                             amsgrad=False)  # epsilon=None, default used here instead  1.0 or 0.1
        # amsgrad=False https://openreview.net/pdf?id=ryQu7f-RZ
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimiser_name,))

    if loss_name == 'ce_jaccard_loss':
        loss_func = jaccard_coef_loss
    elif loss_name == 'iou_loss':
        loss_func = iou_loss
    elif loss_name == 'binary_focal_loss':
        loss_func = binary_focal_loss()
    elif loss_name == 'dice_coef_loss':
        loss_func = dice_coef_loss
    elif loss_name == 'crossentropy_dice_loss':
        loss_func = crossentropy_dice_loss
    else:
        raise ValueError("unexpectedloss_func: %r" % (loss_name,))
    model.compile(
        # optimizer=tf.keras.optimizers.Nadam(lr=1e-2),  # this LR is overriden by base cycle LR if CyclicLR callback used
        optimizer=optimiser,  # this LR is overriden by base cycle LR if CyclicLR callback used
        # loss=dice_coef_loss,
        # metrics=dice_score,
        # loss="binary_crossentropy",
        # metrics=metrics
        loss=loss_func,
        metrics=['accuracy',
                 jaccard_coef_int,
                 dice_coef,
                 "binary_crossentropy",
                 ]
    )

    train_generator = ImageGenerator(scaled_train_X, train_Y, dim=(shape[0], shape[1]),batch_size=batch_size, n_channels=shape[2])
    val_generator = ImageGenerator(scaled_val_X, val_Y, dim=(shape[0], shape[1]), n_channels=shape[2])
    # train_generator = DataGenerator_segmentation(train, dimension=(height, width),
    #                                              batch_size=batch_size,
    #                                              n_channels=n_channels)
    # val_generator = DataGenerator_segmentation(val, dimension=(height, width),
    #                                            batch_size=batch_size,
    #                                            n_channels=n_channels)
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_Y) // batch_size, shuffle=True,
                        epochs=n_epochs,
                        verbose=1,
                        validation_data=val_generator, callbacks=[
            #early_stopping,
            reduce_lr,
            checkpoint,
            tensorboard,  # log metrics
            terminate_nan, #exploding gradients termination
            hp.KerasCallback(run_dir, hparams),  # log hparams
        ],
                        )
    print(history.history['val_dice_coef'][-1])
    print(history.history['dice_coef'][-1])
    # dice_list.append(history.history['dice_coef'][-1])
    # dice_val_list.append(history.history['val_dice_coef'][-1])
    # jaccard_list.append(history.history['jaccard_coef_int'][-1])
    # jaccard_val_list.append(history.history['val_jaccard_coef_int'][-1])

    return history.history['val_accuracy'][-1], history.history['val_dice_coef'][-1], \
           history.history['val_jaccard_coef_int'][-1], history, model


# CHANGEME
n_epochs = 200
tf.summary.experimental.set_step(True)
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001]))
HP_OPTIMIZER = hp.HParam('optimiser', hp.Discrete(['rmsprop']))  # ,['adam','nadam','sgd','rmsprop']
HP_NORM = hp.HParam('norm_name', hp.Discrete(['naive']))  # 'max','naive',
LOSS = hp.HParam('loss', hp.Discrete( ['crossentropy_dice_loss']))  # 'crossentropy_dice_loss','ce_jaccard_loss','iou_loss','binary_focal_loss','dice_coef_loss'
BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1]))  # 1,2,3,4,5,6,10,15,25,30 between 1 and 2 for best 1,2,4,6,10,16,20
PATCH_SIZE = hp.HParam('patch_size', hp.Discrete([64]))  # 256,128,64,32
ACTIVATION = hp.HParam('activation_name', hp.Discrete(['elu']))  # ['elu','relu','gelu','selu','tanh']? Leaky Relu needs to be implemented as a separate layer :( PRelu also does https://tensorlayer.readthedocs.io/en/latest/modules/layers.html#prelu-layer
INITIALISATION = hp.HParam('init_name', hp.Discrete(['he_normal'])) #['he_normal', 'he_uniform', 'glorot_normal','glorot_uniform','random_normal','random_uniform']
METRIC_BCE = 'binary_crossentropy'
METRIC_ACCURACY = 'accuracy'
METRIC_DICE = 'dice_coef'
METRIC_IOU = 'jaccard_coef_int'
METRIC_LOSS = 'loss'

hp.hparams_config(
    hparams=[LOSS, BATCH_SIZE, HP_OPTIMIZER, HP_NORM, HP_LEARNING_RATE, PATCH_SIZE, ACTIVATION,INITIALISATION],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'), hp.Metric(METRIC_DICE, display_name='Dice'),
             hp.Metric(METRIC_IOU, display_name='IoU'), hp.Metric(METRIC_BCE, display_name='BCE'),
             hp.Metric(METRIC_LOSS, display_name='Loss')],
)


def run(run_dir, hparams, name):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy, dice, iou, history, model = train_test_model(hparams, run_dir, name, n_epochs=n_epochs)
        # converting to tf scalar
        # accuracy = tf.reshape(tf.convert_to_tensor(accuracy), []).numpy()
        # dice = tf.reshape(tf.convert_to_tensor(dice), []).numpy()
        # iou = tf.reshape(tf.convert_to_tensor(iou), []).numpy()
    return accuracy, history, model


# plot_logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# plt_file_writer = tf.summary.create_file_writer(plot_logdir)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_metric(history, metric, name):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(fr'C:\Users\leo__\PycharmProjects\Perma_Thesis\plots_{experiment_folder}\{name}_{metric}_curve.png')


from csv import writer
import pickle

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


# runner
session_num = 0

log_dir = f'logs/hparam_tuning_loss_{experiment_folder}/'
for norm_name in HP_NORM.domain.values:
    for optimiser in HP_OPTIMIZER.domain.values:
        for loss_func in LOSS.domain.values:
            for batch_size in BATCH_SIZE.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    for patch_size in PATCH_SIZE.domain.values:
                        for activation_name in ACTIVATION.domain.values:
                            for init_name in INITIALISATION.domain.values:
                                hparams = {
                                    LOSS: loss_func,
                                    BATCH_SIZE: batch_size,
                                    HP_OPTIMIZER: optimiser,
                                    HP_NORM: norm_name,
                                    HP_LEARNING_RATE: learning_rate,
                                    PATCH_SIZE: patch_size,
                                    ACTIVATION: activation_name,
                                    INITIALISATION: init_name
                                }
                                run_name = "run-%d" % session_num
                                name = f'{norm_name}_{batch_size}_{loss_func}_{optimiser}_{learning_rate}_{patch_size}_{activation_name}_{init_name}_{n_epochs}planet'
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})

                                accuracy, history, model = run(log_dir + run_name, hparams, name)
                                hist_df = pd.DataFrame(history.history)
                                file_name = fr'C:\Users\leo__\PycharmProjects\Perma_Thesis\history_files_{experiment_folder}\history_{name}.csv'
                                hist_df.to_csv(file_name)
                                # ### Save model
                                export_path = fr'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_{experiment_folder}\model_{name}'
                                model.save(export_path)
                                file_writer = tf.summary.create_file_writer(log_dir + run_name)
                                file_writer.set_as_default()
                                print("Model exported to: ", export_path)
                                acc_fig = plot_metric(history, "jaccard_coef_int", name)
                                acc_fig = plot_metric(history, "dice_coef", name)
                                loss_fig = plot_metric(history, "loss", name)
                                # test_generator_gt = DataGenerator_segmentation(test, dimension=(patch_size, patch_size),
                                #                                                batch_size=len(test),
                                #                                                n_channels=4)
                                test_generator_gt = ImageGenerator(scaled_test_X, test_Y, dim=(shape[0], shape[1]),
                                                         n_channels=shape[2],batch_size=len(test_Y))
                                test_gt = test_generator_gt.__getitem__(0)
                                score = model.evaluate(test_gt[0], test_gt[1], verbose=1)
                                tf.summary.scalar(METRIC_ACCURACY, score[1])
                                tf.summary.scalar(METRIC_DICE, score[3])
                                tf.summary.scalar(METRIC_IOU, score[2])
                                tf.summary.scalar(METRIC_LOSS, score[0])
                                tf.summary.scalar(METRIC_BCE, score[4])
                                key = ['loss', 'accuracy', 'jaccard_coef_int'
                                    , 'dice_coef', 'binary_crossentropy']
                                append_list_as_row(file_name, key)
                                append_list_as_row(file_name, score)

                                session_num += 1

print(history.history.keys())
print(log_dir)

norm_method = 'naive'
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
# model_path =r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_recontruct_2\model_z_score_4_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_uniform_200'
# model_path =r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_recontruct_3\model_z_score_4_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_uniform_200'
# model_path =r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_z_score_selection\model_z_score_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_uniform_200'
# #model_path =r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_files_adam_new_data_z_score_batch_selection\model_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_gelu_he_normal_200'
#model_path='model_files_adam_new_data_recontruct_naive_balanced\model_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200from_scratch'
model_path='model_files_adam_new_data_naive_planetary\model_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200'
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
print(log_dir)