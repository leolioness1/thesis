#!/usr/bin/env python
# coding: utf-8
import os
from glob import glob
import rasterio.mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf
import tempfile
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_generator import DataGenerator
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
LOCAL_PATH= '../Perma_Thesis/MSI/'

# ### Modelling Pipeline (Test)
IMG_HEIGHT=256
IMG_WIDTH=256
BATCH_SIZE = 3
# ##### Load image, preprocess, augment through image data generator

files = glob(LOCAL_PATH + "**/*.tif")

df_dataset = pd.DataFrame()
df_dataset['image_paths'] = files
df_dataset['labels_string'] = df_dataset['image_paths'].str.split('\\').str[-2]
df_dataset['label'] =  df_dataset['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_dataset = df_dataset.sample(frac=1).reset_index(drop=True) #Randomize

df_dataset

VAL_LOCAL_PATH= '../Perma_Thesis/RGB-thawslump-UTM-Images/batagay/'

df_val = pd.DataFrame()
files_val = glob(VAL_LOCAL_PATH + "**/*.tif")
df_val['image_paths'] = files_val
df_val['labels_string'] = df_val['image_paths'].str.split('\\').str[-2]
df_val['label'] =  df_val['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_val=df_val.sample(frac=1).reset_index(drop=True) #Randomize
df_val
train, test = train_test_split(df_dataset, test_size=0.2, stratify=df_dataset['label'], random_state=123)

train.describe()

test.describe()

#Custom data generator that replaces PIL function on image read of tiff record with rasterio 
#as default Imagedatagenertator seems to be throwing error
from data_generator_segmentation import DataGenerator_segmentation
# #### Custom data generator
train_generator = DataGenerator_segmentation(train, dimension=(256, 256),
                 n_channels=4)
test_generator = DataGenerator_segmentation(test,dimension=(256, 256),
                 n_channels=4)
#Add code for resize
#Add code for normalize range to 0-1
#Add code fro augmentations
train_generator.__getitem__(1)
tf.config.list_physical_devices('GPU')

#encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)

import mlflow
experiment_name = 'Baseline:  UNET No Transfer Learning'
mlflow.set_experiment(experiment_name)
#mlflow.tensorflow.autolog()
height = 256
width = 256
n_channels =4

# Sizes of the training and evaluation datasets.
TRAIN_SIZE = 300
EVAL_SIZE = 100

# Specify model training parameters.
BATCH_SIZE = 16
EPOCHS = 10
BUFFER_SIZE = 2000
OPTIMIZER = 'SGD'
LOSS = 'MeanSquaredError'
METRICS = ['RootMeanSquaredError']

smooth = 1e-12
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


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

def dice_coef_loss(y_true, y_pred):
    """
    Arguments:
        y_true: (string) ground truth image mask
        y_pred : (int) predicted image mask

    Returns:
        Calculated Dice coeffecient loss
    """
    return 1 - dice_coef(y_true, y_pred)

def dice_score(y_true, y_pred, smooth=1, threshold = 0.6):
    """
    Arguments:
        y_true: (string) ground truth image mask
        y_pred : (int) predicted image mask
        smooth : (float) smoothening to prevent divison by 0
        threshold : (float) threshold over which pixel is conidered positive

    Returns:
        Calculated Dice coeffecient for evaluation metric
    """
    y_true_f = K.flatten(y_true)
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), threshold), 'float32')
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth)/ (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
#
# inputs = Input(shape=(height, width, n_channels), name="MSI")
# #### Load Pretrained model - MobileNet / Efficientnet

def conv_block(input_tensor, num_filters):
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
	encoder = layers.Activation('relu')(encoder)
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
	encoder = layers.Activation('relu')(encoder)
	return encoder

def encoder_block(input_tensor, num_filters):
	encoder = conv_block(input_tensor, num_filters)
	encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
	return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
	decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
	decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.Activation('relu')(decoder)
	return decoder

def get_model():
	inputs = layers.Input(shape=(256, 256, n_channels), name="MSI")# 256
	encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
	encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
	encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
	encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
	encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
	center = conv_block(encoder4_pool, 1024) # center
	decoder4 = decoder_block(center, encoder4, 512) # 16
	decoder3 = decoder_block(decoder4, encoder3, 256) # 32
	decoder2 = decoder_block(decoder3, encoder2, 128) # 64
	decoder1 = decoder_block(decoder2, encoder1, 64) # 128
	decoder0 = decoder_block(decoder1, encoder0, 32) # 256
	outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

	model = models.Model(inputs=[inputs], outputs=[outputs])

	# model.compile(
	# 	optimizer=optimizers.get(OPTIMIZER),
	# 	loss=losses.get(LOSS),
	# 	metrics=[metrics.get(metric) for metric in METRICS])
	# model.compile(
	# 	optimizer=tf.keras.optimizers.Adam(
	# 		learning_rate=1e-3
	# 	),  # this LR is overriden by base cycle LR if CyclicLR callback used
	# 	loss=dice_coef_loss,
	# 	metrics=dice_score,
	# )
	return model

	model.compile(
		optimizer=tf.keras.optimizers.Nadam(lr=1e-2),  # this LR is overriden by base cycle LR if CyclicLR callback used
		# loss=dice_coef_loss,
		# metrics=dice_score,
		# loss="binary_crossentropy",
		# metrics=metrics
		loss=jaccard_coef_loss,
		metrics=['binary_crossentropy', jaccard_coef_int]
	)
m = get_model()

m.summary()

# # Load a trained model. 50 epochs. 25 hours. Final RMSE ~0.08.
# MODEL_DIR = ''
# m = tf.keras.models.load_model(MODEL_DIR)
# m.summary()

params = {"height": height
    , "width": width
    ,"n_channels": n_channels
    ,"normalisation": ">1000/1000",
     "model": "UNET"}


with mlflow.start_run() as run:
    mlflow.log_params(params)

# history = m.fit(
#     x=train_generator,
#     epochs=EPOCHS,
#     steps_per_epoch=int(TRAIN_SIZE / BATCH_SIZE),
#     validation_data=test_generator,
#     validation_steps=EVAL_SIZE)
tf.config.list_physical_devices('GPU')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1, mode='min')

history = m.fit(train_generator,
steps_per_epoch=250//3,shuffle=True,
epochs=30,
verbose=1,
validation_data = test_generator,callbacks=[early_stopping, reduce_lr])

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history.csv')
# ### Save model
export_path = tf.saved_model.save(m, 'keras_export')
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

# Log as MLflow artifact
with tempfile.TemporaryDirectory() as temp_dir:
    image_path = os.path.join(temp_dir, "loss_curve.png")
    plt.savefig(image_path)
    mlflow.log_artifact(image_path)

val_generator = DataGenerator_segmentation(df_val, dimension=(256, 256),
                 n_channels=4, to_fit=False)
x=val_generator.__getitem__(1)
print('Running predictions...')
predictions = m.predict(val_generator, verbose=1)
print(predictions[0])

plt.imshow(predictions[-1])
plt.show()
