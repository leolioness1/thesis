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
from data_generator import DataGenerator

LOCAL_PATH= '../Perma_Thesis/MSI/'

# ### Modelling Pipeline (Test)
IMG_HEIGHT=256
IMG_WIDTH=256
BATCH_SIZE = 8
# ##### Load image, preprocess, augment through image data generator


files = glob(LOCAL_PATH + "**/*.tif")
df_dataset = pd.DataFrame()
df_dataset['image_paths'] = files
df_dataset['labels_string'] = df_dataset['image_paths'].str.split('\\').str[-2]
df_dataset['label'] =  df_dataset['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
df_dataset = df_dataset.sample(frac=1).reset_index(drop=True) #Randomize

df_dataset


# VAL_LOCAL_PATH= '../Perma_Thesis/RGB-thawslump-UTM-Images/batagay/'
#
# df_val = pd.DataFrame()
# files_val = glob(VAL_LOCAL_PATH + "**/*.tif")
# df_val['image_paths'] = files_val
# df_val['labels_string'] = df_val['image_paths'].str.split('\\').str[-2]
# df_val['label'] =  df_val['labels_string'].apply(lambda x: 1 if x == 'thaw' else 0)
# df_val=df_val.sample(frac=1).reset_index(drop=True) #Randomize
# df_val


print("Full Dataset label distribution")
print(df_dataset.groupby('labels_string').count())

train, val = train_test_split(df_dataset, test_size=0.3, stratify=df_dataset['label'], random_state=123)
val, test = train_test_split(val, test_size=0.1, stratify=df_dataset['label'], random_state=123)
print("\nTrain Dataset label distribution")
print(train.groupby('labels_string').count())
print("\nVal Dataset label distribution")
print(val.groupby('labels_string').count())
print("\nTest Dataset label distribution")
print(test.groupby('labels_string').count())
n_channels =3
#Custom data generator that replaces PIL function on image read of tiff record with rasterio 
#as default Imagedatagenertator seems to be throwing error
# #### Custom data generator
train_generator = DataGenerator(train, dimension=(256, 256), n_channels=n_channels)
val_generator = DataGenerator(val,dimension=(256, 256),
                 n_channels=n_channels)
#Add code for resize
#Add code for normalize range to 0-1
#Add code fro augmentations
train_generator.__getitem__(1)
tf.config.list_physical_devices('GPU')
#encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)

import mlflow
experiment_name = 'Baseline: Transfer Learning'
mlflow.set_experiment(experiment_name)
# mlflow.tensorflow.autolog()
# height = 256
# width = 256
# n_channels =4
#
# # create the base pre-trained model try adapt to take 4 channels
# base_model = MobileNetV2( weights='imagenet', include_top=False)
# input_tensor = Input(shape=(height,width,n_channels))
# conv= tf.keras.layers.Conv2D(3,(3,3),padding="same",activation="relu")(input_tensor)
# out = base_model(conv)
#
# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(out)
# # let's add a fully-connected layer
# x = Dense(128, activation='relu')(x)
# # and a logistic layer -- let's say we have 200 classes
# predictions = Dense(1, activation='sigmoid')(x)
#
# # this is the model we will train
# model = Model(inputs=input_tensor, outputs=predictions)

# this is the model we will train  with 3 channels only
# #### Load Pretrained model - MobileNet / Efficientnet
height = 256
width = 256
n_channels =3

inputs = Input(shape=(height, width, n_channels), name="MSI")
# #### Load Pretrained model - MobileNet / Efficientnet
# create the base pre-trained model
base_model = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(128, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

params = {"height": height
    , "width": width
    ,"n_channels": n_channels
    ,"normalisation": ">1000/1000"}
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),  # this LR is overriden by base cycle LR if CyclicLR callback used
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1, mode='min')

train_generator = DataGenerator(train, dimension=(height, width),
                 n_channels=n_channels)
test_generator = DataGenerator(test,dimension=(height, width),
                 n_channels=n_channels)


with mlflow.start_run() as run:
    mlflow.log_params(params)

    history = model.fit(train_generator,
    steps_per_epoch=424//6,shuffle=True,
    epochs=20,
    verbose=1,
    validation_data = val_generator,callbacks=[
            #early_stopping,
             reduce_lr]
                        )

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

    # Log as MLflow artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "loss_curve.png")
        plt.savefig(image_path)
        mlflow.log_artifact(image_path)



test_generator = DataGenerator(test, dimension=(256, 256),
                 n_channels=3, to_fit=False)
x= test_generator.__getitem__(1)
print('Running predictions...')
predictions = model.evaluate(test_generator, verbose=1)
print(predictions[0])

predictions_bol= predictions >0.5