from tensorflow.keras.layers import Lambda, concatenate, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, \
    GlobalAveragePooling2D, Conv2DTranspose
from tensorflow.keras.metrics import MeanIoU as mean_iou
from tensorflow.keras.models import Model

from tensorflow.keras.losses import binary_crossentropy
# !/usr/bin/env python
# coding: utf-8
import io
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
from glob import glob
import rasterio
import rasterio.mask
import geopandas as gpd
import fiona
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import tempfile

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, UpSampling2D, \
    BatchNormalization, Activation, Add, Multiply, Dropout, Lambda, MaxPooling2D, concatenate, Dense, Reshape
from tensorflow.keras.initializers import glorot_normal, he_normal, glorot_uniform, he_uniform
from tensorflow.keras.applications import MobileNetV2
# train the model on the new data for a few epochs
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# #### Custom data generator
LOCAL_PATH= '../Perma_Thesis/output_windows'
df_dataset = pd.DataFrame()
files = glob(LOCAL_PATH + "**/*.tiff")
df_dataset['image_paths'] = files
# df_dataset['labels_string'] = df_dataset['image_paths'].str.split('\\').str[-2]
# df_dataset['label'] =  df_dataset['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_dataset = df_dataset.sample(frac=1).reset_index(drop=True) #Randomize
df_dataset['image_paths'].str.split('/').str[-2]
df_dataset




print("Full Dataset label distribution")
print(df_dataset.count())

train, val = train_test_split(df_dataset, test_size=0.3, random_state=123)
val, test = train_test_split(val, test_size=0.1, random_state=123)
print("\nTrain Dataset label distribution")
print(train.count())
print("\nVal Dataset label distribution")
print(val.count())
print("\nTest Dataset label distribution")
print(test.count())

# Custom data generator that replaces PIL function on image read of tiff record with rasterio
# as default Imagedatagenertator seems to be throwing error

from data_generator_segmentation_adam_data import DataGenerator_segmentation
smooth = 1e-12
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    tf.summary.scalar('jaccard_coef', data=K.mean(jac))
    return K.mean(jac)


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

import dill

from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
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
    tf.summary.scalar('dice_coef', data=(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# wrong
# def dice_coef_2(y_true, y_pred, smooth=1):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

# def dice_coeff(y_true, y_pred):
#     smooth = 1.0
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def crossentropy_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# bad
#
# def dice_coef_loss(y_true, y_pred):
#     return 1-dice_coef_2(y_true, y_pred)

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


def get_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=4, activation_func='elu'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
    # t1 = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2,0.2, 'nearest', interpolation = 'bilinear')(inputs)
    # t2 = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.3, 0.3, 'nearest', interpolation='bilinear')(t1)
    c1 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c5)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation=activation_func, kernel_initializer='he_normal', padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, mean_iou(2)])
    return model


# # Load a trained model. 50 epochs. 25 hours. Final RMSE ~0.08.
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


def train_test_model(hparams, run_dir, n_epochs=5):
    # 5. Set the callbacks for saving the weights and the tensorboard
    # + "/lr_{}".format(round(learning_rate,8))
    #    height = hparams[PATCH_SIZE]
    #    width = hparams[PATCH_SIZE]
    height, width = 64, 64
    n_channels = 4
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=run_dir, histogram_freq=0, update_freq="epoch")
    model = get_unet(IMG_WIDTH=height, IMG_HEIGHT=width, IMG_CHANNELS=n_channels, activation_func='elu')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1, mode='min')
    # optimiser_name= hparams[HP_OPTIMIZER]
    optimiser_name = 'adam'
    loss_name = hparams[LOSS]
    batch_size = hparams[BATCH_SIZE]
    # loss_list.append(loss_name)
    # batch_list.append(batch_size)
    learning_rate = 0.0001
    now_time = datetime.now().strftime("%Y%m%d%H%M%S")
    weights_path = fr"C:\Users\leo__\PycharmProjects\Perma_Thesis\weights_files\adam_data_{now_time}_batch_{round(batch_size, 8)}_{loss_name}.hdf5"
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)
    # 4. Select the optimizer and the learning rate (default option is Adam)
    if optimiser_name == 'rmsprop':
        optimiser = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    elif optimiser_name == 'nadam':
        optimiser = tf.keras.optimizers.Nadam(learning_rate=learning_rate)  # learning_rate=learning_rate
    elif optimiser_name == 'sgd':
        optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.0)
    elif optimiser_name == 'adam':
        optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                             decay=0.0, amsgrad=False)
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimiser_name,))

    if loss_name == 'jaccard_loss':
        loss_func = jaccard_coef_loss
    elif loss_name == 'binary_focal_loss':
        loss_func = binary_focal_loss()
    elif loss_name == 'dice_loss':
        loss_func = dice_loss
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
                 "binary_crossentropy"
                 ]
    )

    train_generator = DataGenerator_segmentation(train, dimension=(height, width), batch_size=batch_size,
                                                 n_channels=n_channels)
    val_generator = DataGenerator_segmentation(val, dimension=(height, width), batch_size=batch_size,
                                               n_channels=n_channels)
    history = model.fit(train_generator,
                        steps_per_epoch=len(train) // batch_size, shuffle=True,
                        epochs=n_epochs,
                        verbose=1,
                        validation_data=val_generator, callbacks=[
            #            early_stopping,
            reduce_lr,
            checkpoint,
            tensorboard,  # log metrics
            hp.KerasCallback(run_dir, hparams),  # log hparams
        ],
                        )
    print(history.history['val_accuracy'][-1])
    print(history.history['accuracy'][-1])
    # dice_list.append(history.history['dice_coef'][-1])
    # dice_val_list.append(history.history['val_dice_coef'][-1])
    # jaccard_list.append(history.history['jaccard_coef_int'][-1])
    # jaccard_val_list.append(history.history['val_jaccard_coef_int'][-1])

    return history.history['val_accuracy'][-1], history, model


tf.summary.experimental.set_step(True)
# HP_LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([0.0001]))
HP_OPTIMIZER = hp.HParam('optimiser', hp.Discrete(['adam']))
LOSS = hp.HParam('loss', hp.Discrete(['dice_loss','crossentropy_dice_loss']))
BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([3,4]))
# PATCH_SIZE= hp.HParam('patch_size', hp.Discrete([64]))
METRIC_BCE = 'binary_crossentropy'
METRIC_ACCURACY = 'accuracy'
hp.hparams_config(
    hparams=[LOSS, BATCH_SIZE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
)

n_epochs = 10


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy, history, model = train_test_model(hparams, run_dir, n_epochs=n_epochs)
        # converting to tf scalar
        accuracy = tf.reshape(tf.convert_to_tensor(accuracy), []).numpy()
        tf.summary.scalar(METRIC_ACCURACY, accuracy)
        # tf.summary.scalar(METRIC_BCE, bce, step=1)
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


def plot_metric(history, metric, patch_size, optimiser):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()
    plt.savefig(f"adam_data_{patch_size}_{optimiser}_{metric}.png")


# runner
session_num = 0

log_dir = 'logs/hparam_tuning_adam_data/'
for loss_func in LOSS.domain.values:
    for batch_size in BATCH_SIZE.domain.values:
        hparams = {
            LOSS: loss_func,
            BATCH_SIZE: batch_size,

            # HP_LEARNING_RATE: learning_rate
        }
        patch_size = 64
        learning_rate = 0.0001
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        accuracy, history, model = run(log_dir + run_name, hparams)
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(f'history_{batch_size}_{loss_func}_{run_name}_{n_epochs}.csv')
        # ### Save model
        export_path = f'model_adam_data_{batch_size}_{loss_func}_{run_name}_{n_epochs}.h5'
        model.save(export_path)
        print("Model exported to: ", export_path)
        file_writer = tf.summary.create_file_writer(log_dir + run_name)
        file_writer.set_as_default()
        test_generator_gt = DataGenerator_segmentation(test, dimension=(patch_size, patch_size),
                                                       batch_size=10,
                                                       n_channels=4)
        test_gt = test_generator_gt.__getitem__(0)
        print('Running predictions...')
        score = model.evaluate(test_gt[0], test_gt[1], verbose=1)
        predictions = model.predict(test_gt[0], verbose=1)
        print(predictions[0])
        print(score)
        list = []
        for i in range(0, len(predictions)):
            list.append(np.max(predictions[i]))
        # plot_sample(i)
        print(list)
        file_writer = tf.summary.create_file_writer(log_dir + run_name)
        file_writer.set_as_default()
        acc_fig = plot_metric(history, "accuracy", batch_size, loss_func)
        tf.summary.image(f"{batch_size}_{loss_func}_{run_name}_accuracy_curve", plot_to_image(acc_fig))
        acc_fig = plot_metric(history, "dice_coef", batch_size, loss_func)
        tf.summary.image(f"{batch_size}_{loss_func}_{run_name}_dice_curve", plot_to_image(acc_fig))
        loss_fig = plot_metric(history, "loss", batch_size, loss_func)
        tf.summary.image(f"{batch_size}_{loss_func}_{run_name}_loss_curve", plot_to_image(loss_fig))
        session_num += 1

print(history.history.keys())
test_generator_gt =DataGenerator_segmentation(test,dimension=(64,64),batch_size=10, n_channels=4)
# test_item=test_generator.__getitem__(0)

test_gt = test_generator_gt.__getitem__(0)
test_gt[0][-1].shape
test_gt[1][0].shape
x_example_ex = np.expand_dims(test_gt[0][0], axis=0)
mask_example_ex = np.expand_dims(test_gt[1][0], axis=0)
print('Running predictions...')

model_path=r'C:\Users\leo__\PycharmProjects\Perma_Thesis\model_3_crossentropy_dice_loss_run-2_50'
reconstructed_model = tf.keras.models.load_model(model_path, compile=False)
reconstructed_model.compile(optimizer='adam',
                       loss=crossentropy_dice_loss,
                       metrics=['accuracy',
                 jaccard_coef_int,
                 dice_coef,
                 dice_coeff,
                 "binary_crossentropy"
                 ])
predictions = reconstructed_model.predict(test_gt[0], verbose=1)
score = reconstructed_model.evaluate(test_gt[0], test_gt[1], verbose=1)
print(predictions[0])
print(score)


def plot_sample(ix=None):
    """Function to plot the results"""

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))  # 4
    ax[0].imshow(test_gt[0][ix])

    ax[0].contour(test_gt[1][ix].squeeze(), colors='w', levels=[0.5])
    ax[0].set_title('Image')

    ax[1].imshow(test_gt[1][ix].squeeze())
    ax[1].set_title('Mask')

    ax[2].imshow(predictions[ix].squeeze(), vmin=0, vmax=1)

    ax[2].contour(test_gt[1][ix].squeeze(), colors='w', levels=[0.5])
    ax[2].set_title('Predicted')


plt.show()
list = []
for i in range(0, len(predictions)):
    list.append(np.max(predictions[i]))
    plot_sample(i)
print(list)

# def plot_sample(ix=None):
#     """Function to plot the results"""
#     fig, ax = plt.subplots(10, 3, figsize=(10, 10)) #4
#     for ix in range(0, len(predictions)):
#
#         ax[ix][0].imshow(test_gt[0][ix])
#
#         ax[ix][0].contour(test_gt[1][ix].squeeze(), colors='w', levels=[0.5])
#         ax[ix][0].set_title('Image')
#
#         ax[ix][1].imshow(test_gt[1][ix].squeeze())
#         ax[ix][1].set_title('Mask')
#
#         ax[ix][2].imshow(predictions[ix].squeeze(), vmin=0, vmax=1)
#
#         ax[ix][2].contour(test_gt[1][ix].squeeze(), colors='w', levels=[0.5])
#         ax[ix][2].set_title('Predicted')

# cfd_last = x_cfd.shape[0]
#
# for i in range(10): # number of rows of plot
#     for j in range(5):
#
#         if i*5+j == cfd_last:
#             break
#
#         pixels=cp_cfd[i*5+j,:].reshape(79, 14)
#         plt.subplot(1,5,j+1)
#         plt.imshow(pixels, cmap=plt.cm.jet)
#         plt.title('#'+str(i*5+j+1))
#
#
#     plt.show()
