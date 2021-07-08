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
import tempfile
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# #### Setting paths
# GCS_IMAGE_PATH = 'gs://acn-cloudwars3-permafrost-training-data/RGB-thawslump-UTM-Images/'
# VALIDATED_SHAPE_FILE_PATH = 'gs://acn-cloudwars3-permafrost-training-data/shapefiles/merged_thaw_slumps_validated/merged_thaw_slumps_validated.shp'
# ALL_SHAPE_FILE_PATH = 'gs://acn-cloudwars3-permafrost-training-data/shapefiles/Thaw slumps_first iteration/merged_thaw_slumps.shp'

LOCAL_PATH= '../Perma_Thesis/MSI/'
#LOCAL_PATH= './test_128'

#!gsutil ls gs://acn-cloudwars3-permafrost-training-data/test_128/*.tif
#!gsutil -m cp -r gs://acn-cloudwars3-permafrost-training-data/test_128 ./

# #### Copying from bucket to VM

#!gsutil cp -r gs://acn-cloudwars3-permafrost-training-data/RGB-thawslump-UTM-Images/MSI ./


# #### Explore n view .tif

#src = rasterio.open('./RGB-thawslump-UTM-Images/control/3-3.tif')
src = rasterio.open('../Perma_Thesis/MSI/thaw/25-20190905_195023_1032.tif')
src = rasterio.open('../Perma_Thesis/RGB-thawslump-UTM-Images/thaw/0-20190710_214246_1001.tif')



# src.crs
# array = src.read()
# array.shape
#
# img_temp = array[:4,:256,:256]
# img_temp = img_temp/1000 # Hack but need to figure out correct value
# img_final = np.moveaxis(img_temp, 0, -1)
#
# np.moveaxis(array, 0, -1).shape # Test moving channel to end
#
#
# img_final.shape
#
# plt.imshow(img_final)
# plt.show()
#
# # Proves that crop to size 40*40 is better but maybe make it 100*100??
# img_cropped = img_final[44:84, 44:84,:]
# plt.imshow(img_cropped)
# plt.show()
# #but still images are very bad quality resolution and need to add more bands!!!!



# array = array[:,98:160,98:160]
# fig, ax = plt.subplots(4,2)
# ax[0][0].imshow(array[0]);
# ax[0][1].imshow(array[1]);
# ax[1][0].imshow(array[2]);
# ax[1][1].imshow(array[3]);
# ax[2][0].imshow(array[4]);
# ax[2][1].imshow(array[5]);
# ax[3][0].imshow(array[6]);
#
#
# np.nan_to_num(array[6], nan=0,copy=False)
#
#
#
# np.unique(array[3], return_counts=True) # Should make it 0 when using as mask


# #### Changing CRS of shape files (Not required anymore)

#
# validated_shape_files = gpd.read_file(VALIDATED_SHAPE_FILE_PATH)
# validated_shape_files = validated_shape_files.to_crs("EPSG:32604")
# shapes = validated_shape_files['geometry']
#
# shapes


# #### Find overlap of tif file and validated thaw slump polygon to build masks (Not required anymore)


#out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
#out_meta = src.meta

#Looks like the polygon shapes do not overlap with tif files even after changing CRS.
#Something wrong here. Check!


# ### Modelling Pipeline (Test)


IMG_HEIGHT=256
IMG_WIDTH=256
BATCH_SIZE = 8


# ##### Load image, preprocess, augment through image data generator

# #### Custom data generator

files = glob(LOCAL_PATH + "**/*.tif")
df_dataset = pd.DataFrame()
df_dataset['image_paths'] = files
df_dataset['labels_string'] = df_dataset['image_paths'].str.split('\\').str[-2]
df_dataset['label'] =  df_dataset['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_dataset = df_dataset.sample(frac=1).reset_index(drop=True) #Randomize

df_dataset

train, test = train_test_split(df_dataset, test_size=0.2, stratify=df_dataset['label'], random_state=123)

train.describe()

test.describe()

#Custom data generator that replaces PIL function on image read of tiff record with rasterio 
#as default Imagedatagenertator seems to be throwing error

from data_generator import DataGenerator
from data_generator_segmentation import DataGenerator_segmentation
train_generator = DataGenerator(train, dimension=(256, 256),
                 n_channels=4)
test_generator = DataGenerator(test,dimension=(256, 256),
                 n_channels=4)

#Add code for resize
#Add code for normalize range to 0-1
#Add code fro augmentations
train_generator.__getitem__(1)


def build_keras_model(inputShape):
    inputs = keras.Input(shape=inputShape)

    x = Conv2D(3, (3,3), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu', name='l2-layer')(x)
    l2_logits = x
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2)(x)

    model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='dct_model')
    return model

IMG_HEIGHT=256
IMG_WIDTH=256
model = build_keras_model((IMG_HEIGHT, IMG_WIDTH, 4))
model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),  # this LR is overriden by base cycle LR if CyclicLR callback used
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
height = 256
width = 256
n_channels =3

import mlflow
experiment_name = 'Baseline: Transfer Learning'
mlflow.set_experiment(experiment_name)
mlflow.tensorflow.autolog()

params = {"height": height
    , "width": width
    ,"n_channels": n_channels
    ,"normalisation": ">1000/1000"
     ,"augmentation": "flip h& v, rotation 0.2"}

#flipping, blurring, cropping, and scaling huant w]et al 2020
#Was 256 changed to 40 on 12/03 based on thaw slump size info
def simple_convnet(IMG_HEIGHT=height, IMG_WIDTH=width, N_CHANNELS=n_channels):
    model = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.Conv2D(16,3,padding="same",activation="relu",input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS),),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 2, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 2, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),  # this LR is overriden by base cycle LR if CyclicLR callback used
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    #print(model.summary())

    return model

model = simple_convnet(IMG_HEIGHT=256, IMG_WIDTH=256, N_CHANNELS=n_channels)

tf.config.list_physical_devices('GPU')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1, mode='min')
#Add checkpoints to rollback to best performing model

with mlflow.start_run() as run:
    mlflow.log_params(params)

history = model.fit(train_generator,
steps_per_epoch=424//6,shuffle=True, 
epochs=30,
verbose=1,
validation_data = test_generator,callbacks=[early_stopping, reduce_lr])
#code works. Not enough data to train a model.

# ### Save model
export_path = tf.saved_model.save(model, 'keras_export')
print("Model exported to: ", export_path)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history.csv')

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



