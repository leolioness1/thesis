from tensorflow.keras.layers import Lambda,concatenate,Conv2D, MaxPooling2D, Dropout, Flatten, Dense,Input, GlobalAveragePooling2D,Conv2DTranspose
from tensorflow.keras.metrics import MeanIoU as mean_iou
from tensorflow.keras.models import Model

from tensorflow.keras.losses import binary_crossentropy
#!/usr/bin/env python
# coding: utf-8
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
#from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, UpSampling2D,     BatchNormalization, Activation, Add, Multiply, Dropout, Lambda,MaxPooling2D,concatenate, Dense, Reshape
from tensorflow.keras.initializers import glorot_normal, he_normal, glorot_uniform, he_uniform
from tensorflow.keras.applications import MobileNetV2
# train the model on the new data for a few epochs
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#using just positive image labels
LOCAL_PATH= '../Perma_Thesis/MSI/thaw'

VAL_LOCAL_PATH= '../Perma_Thesis/RGB-thawslump-UTM-Images/batagay/'

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
    img=img_object.read()
    #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack

    mask = img[-1,:256,:256]
    mask_final = np.moveaxis(mask, 0, -1)
    np.nan_to_num(mask_final, nan=0,copy=False)#Change nans from data to 0 for mask
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

# #### Custom data generator

df_dataset = pd.DataFrame()
files = glob(LOCAL_PATH + "**/*.tif")
df_dataset['image_paths'] = files
df_dataset['labels_string'] = df_dataset['image_paths'].str.split('\\').str[-2]
df_dataset['label'] =  df_dataset['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_dataset = df_dataset.sample(frac=1).reset_index(drop=True) #Randomize
df_dataset['image_paths'].str.split('\\').str[-2]
df_dataset


df_val = pd.DataFrame()
files_val = glob(VAL_LOCAL_PATH + "**/*.tif")
df_val['image_paths'] = files_val
df_val['labels_string'] = df_val['image_paths'].str.split('\\').str[-2]
df_val['label'] =  df_val['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_val=df_val.sample(frac=1).reset_index(drop=True) #Randomize
df_val

print("Full Dataset label distribution")
print(df_dataset.groupby('labels_string').count())

train, test = train_test_split(df_dataset, test_size=0.2, random_state=123)
print("\nTrain Dataset label distribution")
print(train.groupby('labels_string').count())
print("\nVal Dataset label distribution")
print(test.groupby('labels_string').count())
print("\nTest Dataset label distribution")
print(df_val.groupby('labels_string').count())

#Custom data generator that replaces PIL function on image read of tiff record with rasterio
#as default Imagedatagenertator seems to be throwing error

from data_generator_segmentation import DataGenerator_segmentation

height = 256
width = 256
n_channels =4
train_generator = DataGenerator_segmentation(train,batch_size=6, n_channels=n_channels)
test_generator = DataGenerator_segmentation(test,batch_size=6, n_channels=n_channels)

#Add code for resize
#Add code for normalize range to 0-1
#Add code fro augmentations

x_ex=train_generator.__getitem__(1)

import mlflow
experiment_name = 'Baseline Segmentation: elu, he_normal init, nadam'
mlflow.set_experiment(experiment_name)
#mlflow.tensorflow.autolog()


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
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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

def get_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=4):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)


    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, mean_iou(2)])
    return model


# # Load a trained model. 50 epochs. 25 hours. Final RMSE ~0.08.
# MODEL_DIR = ''
# m = tf.keras.models.load_model(MODEL_DIR)
# m.summary()
height = 256
width = 256
n_channels =4

params = {"height": height
    , "width": width
    ,"n_channels": n_channels
    ,"normalisation": ">10000/10000",
     "model": "UNET"}

logdir= 'logs/hparam_tuning'
save_weights_path= "'../" + datetime.now().strftime("%Y%m%d%H%M%S")
save_plots_path="logs/plots/" + datetime.now().strftime("%Y%m%d%H%M%S")

def train_test_model(hparams):
    # 5. Set the callbacks for saving the weights and the tensorboard
    weights = datetime.now().strftime("%Y%m%d%H%M%S") + "_{epoch:02d}.hdf5"  # "_lr_{}".format(round(learning_rate,8))
    checkpoint = ModelCheckpoint(weights, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True)
    logs_dir = logdir  # + "/lr_{}".format(round(learning_rate,8))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0, write_images=False)
    model = get_unet()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1, mode='min')
    optimiser_name= hparams[HP_OPTIMIZER]
    learning_rate = hparams[HP_LEARNING_RATE]
    # 4. Select the optimizer and the learning rate (default option is Adam)
    if optimiser_name == 'rmsprop':
        optimiser = tf.keras.RMSprop(learning_rate=learning_rate,rho=0.9, epsilon=None, decay=0.0)
    elif optimiser_name == 'nadam':
        optimiser = tf.keras.optimizers.Nadam(learning_rate=learning_rate)#learning_rate=learning_rate
    elif optimiser_name == 'sgd':
        optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate,decay=0.0)
    elif optimiser_name == 'adam':
        optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimiser_name,))
    model.compile(
        # optimizer=tf.keras.optimizers.Nadam(lr=1e-2),  # this LR is overriden by base cycle LR if CyclicLR callback used
        optimizer=optimiser,  # this LR is overriden by base cycle LR if CyclicLR callback used
        # loss=dice_coef_loss,
        # metrics=dice_score,
        # loss="binary_crossentropy",
        # metrics=metrics
        loss=jaccard_coef_loss,
        metrics=['accuracy',
                 jaccard_coef_int
                 ]
    )
    history = model.fit(train_generator,
                    steps_per_epoch=250 // 6, shuffle=True,
                    epochs=5,
                    verbose=1,
                    validation_data=test_generator, callbacks=[
#            early_stopping,
            reduce_lr,
            checkpoint,
            tensorboard,  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
      ],
    )
    return history.history['val_accuracy'][-1],history, model

#
# with mlflow.start_run() as run:
#     mlflow.log_params(params)
#
#     history = m.fit(train_generator,
#                     steps_per_epoch=250 // 6, shuffle=True,
#                     epochs=100,
#                     verbose=1,
#                     validation_data=test_generator, callbacks=[
# #            early_stopping,
#             reduce_lr,
#         tf.keras.callbacks.TensorBoard(logdir),  # log metrics
#         hp.KerasCallback(logdir, hparams),  # log hparams
#       ],
#     )
#
#
#     hist_df = pd.DataFrame(history.history)
#     hist_df.to_csv('history.csv')
#     # ### Save model
#     export_path = tf.saved_model.save(m, 'keras_export')
#     print("Model exported to: ", export_path)
#
#     # Removing the first value of the loss
#     losses = history.history['loss']
#     val_losses = history.history['val_loss']
#
#     # Looking at the loss curve
#     plt.plot(losses)
#     plt.plot(val_losses)
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.show()
#
#     # Log as MLflow artifact
#     with tempfile.TemporaryDirectory() as temp_dir:
#         image_path = os.path.join(temp_dir, "loss_curve.png")
#         plt.savefig(image_path)
#         mlflow.log_artifact(image_path)


tf.summary.experimental.set_step(True)
HP_LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([0.001, 0.0005, 0.0001]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'nadam','rmsprop']))
METRIC_BCE = 'binary_crossentropy'
METRIC_ACCURACY= 'accuracy'
# 5. Set the callbacks for saving the weights and the tensorboard
weights = "weights_{epoch:02d}.hdf5"#"_lr_{}".format(round(learning_rate,8))
checkpoint = ModelCheckpoint(weights, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True)
logs_dir = logdir #+ "/lr_{}".format(round(learning_rate,8))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0, write_images=False)
# with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
hp.hparams_config(
    hparams=[HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d%H%M%S")
logs_dir
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

# 5. Set the callbacks for saving the weights and the tensorboard
weights = datetime.now().strftime("%Y%m%d%H%M%S") + "_{epoch:02d}.hdf5"#"_lr_{}".format(round(learning_rate,8))
checkpoint = ModelCheckpoint(weights, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
logs_dir = logdir #+ "/lr_{}".format(round(learning_rate,8))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0, write_images=False)
# log_dir ='\\logs\\fit\\' +datetime.now().strftime('%Y%m%d-%H%M%S')

hp.hparams_config(
hparams=[HP_OPTIMIZER, HP_LEARNING_RATE],
metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
)

def run(run_dir, hparams):
  # with tf.summary.create_file_writer(run_dir).as_default():
  with tf.summary.create_file_writer(run_dir).as_default():
      hp.hparams(hparams)  # record the values used in this trial
      accuracy,history,model = train_test_model(hparams)
      # converting to tf scalar
      accuracy = tf.reshape(tf.convert_to_tensor(accuracy), []).numpy()
      tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

      # tf.summary.scalar(METRIC_BCE, bce, step=1)
  return accuracy,history, model

session_num = 0


for optimiser in HP_OPTIMIZER.domain.values:
    for learning_rate in HP_LEARNING_RATE.domain.values:
      hparams = {
          HP_OPTIMIZER: optimiser,
          HP_LEARNING_RATE: learning_rate
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      accuracy,history, model=run('logs/hparam_tuning/' + run_name, hparams)
      hist_df = pd.DataFrame(history.history)
      hist_df.to_csv('history.csv')
      # ### Save model
      export_path = tf.saved_model.save(model, 'keras_export')
      print("Model exported to: ", export_path)

      # Removing the first value of the loss
      losses = history.history['loss']
      val_losses = history.history['val_loss']

      # Looking at the loss curve
      plt.plot(losses)
      plt.plot(val_losses)
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()
      plt.savefig(f"{run_name}loss_curve.png")
      session_num += 1


def plot_metric(history, metric):
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