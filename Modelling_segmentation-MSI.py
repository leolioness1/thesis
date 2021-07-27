#!/usr/bin/env python
# coding: utf-8
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
n_channels=4
def load_image(image_path):
      """Load grayscale image
      :param image_path: path to image to load
      :return: loaded image
      """
      img_object = rasterio.open(image_path)
      img=img_object.read()
      #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
      size=64
      img_temp = img[:n_channels,:256,:256]
      max_0= 7876
      max_1=8104
      max_2=8072
      max_3=8360

      # img_temp[img_temp > 1000] = 1000
      # img_temp = img_temp / 1000
      # img_temp[0, :256, :256] = img_temp[0, :256, :256] / max_0
      # img_temp[1, :256, :256] = img_temp[1, :256, :256] / max_1
      # img_temp[2, :256, :256] = img_temp[2, :256, :256] / max_2
      # img_temp[3, :256, :256] = img_temp[3, :256, :256] / max_3
      img_final = np.moveaxis(img_temp, 0, -1)
      np.nan_to_num(img_final, nan=0, copy=False)
      mask = img[-1, :256, :256]
      mask_final = np.moveaxis(mask, 0, -1)
      np.nan_to_num(mask_final, nan=0, copy=False)  # Change nans from data to 0 for mask
        #Reducing image size to 40*40 crop from centre based on finding on 12/03 on thaw slump size being avg
        #400m so 40 pixels
      startx = 98 #(128-size/2)
      starty = 98 #(128-size/2)
      img_final = img_final[startx:startx+size,startx:starty+size,:]
      mask_final = mask_final[startx:startx + size, startx:starty + size]
      return img_final, mask_final

def load_mask(image_path):
    img_object = rasterio.open(image_path)
    img=img_object.read()
    #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack

    mask = img[-1,:256,:256]
    mask_final = np.moveaxis(mask, 0, -1)
    np.nan_to_num(mask_final, nan=0,copy=False)#Change nans from data to 0 for mask
    return mask_final

IMAGE_DIR_PATH = '../Perma_Thesis/MSI/thaw/'
BATCH_SIZE = 4

total_64=4096
total_128=128*128
total_256=256*256
std_0=math.sqrt((max_0/total_64)/(313-1))
std_1=math.sqrt((max_1/total_64)/(313-1))
std_2=math.sqrt((max_2/(total_64))/(313-1))
std_3=math.sqrt((max_3/(total_64))/(313-1))
print(std_0,std_1,std_2,std_3)
combined=list(zip(image_paths,val_list))
# create list of PATHS
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.tif')]
from data_generator_segmentation import DataGenerator_segmentation
val_list=[]
max_0= 0
max_1=0
max_2=0
max_3=0
# val_it=[]
# Y = np.empty((len(image_paths), *dimension))
avg_0_m = np.full(
    shape=(64, 64),
    fill_value=avg_0,
    dtype=np.float32
)
avg_1_m = np.full(
    shape=(64, 64),
    fill_value=avg_1,
    dtype=np.float32
)
avg_2_m = np.full(
    shape=(64, 64),
    fill_value=avg_2,
    dtype=np.float32
)
avg_3_m = np.full(
    shape=(64, 64),
    fill_value=avg_3,
    dtype=np.float32
)
for i, ID in enumerate(image_paths):
    print(i, ID)
    image_test,mask_test=load_image(image_path=ID)
    # Initialization

    # Generate data

        # Store sample
        # Y_temp = load_mask(ID)
    # Y[i,] = mask_test

    # mask_test= load_mask(image_path=image)

    # print( "Test Image dimensions:" + str(image_test.shape))
    #
    # print( "Test Mask dimensions:" + str(mask_test.shape))

    val_list.append(tf.math.count_nonzero(mask_test).numpy())
    # i_0 = np.int(np.sum(image_test[:256, :256, 0]))
    # i_1 = np.int(np.sum(image_test[:256, :256, 1]))
    # i_2 = np.int(np.sum(image_test[:256, :256, 2]))
    # i_3 = np.int(np.sum(image_test[:256, :256, 3]))
    # i_0=np.sum(np.square(np.subtract(image_test[:64,:64,0],avg_0_m)))
    # i_1 =np.sum(np.square(np.subtract(image_test[:64,:64,1],avg_1_m)))
    # i_2 = np.sum(np.square(np.subtract(image_test[:64,:64,2],avg_2_m)))
    # i_3 = np.sum(np.square(np.subtract(image_test[:64,:64,3],avg_3_m)))
    # val_it.append(tf.math.count_nonzero(Y[i]).numpy())
    # plt.imshow(image_test)
    # plt.show()
    # max_0=max_0+i_0
    # max_1+=i_1
    # max_2+=i_2
    # max_3+=i_3
    # plt.imshow(mask_test)
    # plt.show()



val_list_perc= np.multiply(np.divide(val_list,4096),100)


import numpy as np
import math
from matplotlib import pyplot as plt

data=val_list

bins = np.linspace(math.ceil(min(data)),
                   math.floor(max(data)),
                   50) # fixed number of bins

plt.xlim([min(data)-5, max(data)+5])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed number of bins)')
plt.xlabel('variable X (20 evenly spaced bins)')
plt.ylabel('count')

plt.show()



import numpy as np
import random
from matplotlib import pyplot as plt

data=val_list
# fixed bin size
bins = np.arange(math.ceil(min(data)),
                   math.floor(max(data)), 50) # fixed bin size

plt.xlim([min(data)-5, max(data)+5])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Distribution of RTS Positive Pixels')
plt.xlabel('RTS positive pixels (bin size = 50)')
plt.ylabel('Count')
plt.show()

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

plt.imshow(x_ex[-1][0])
plt.show()
import mlflow
experiment_name = 'Baseline Segmentation- 128: Jaccard Index Transfer Learning, no early stopping, just thaw imagw'
mlflow.set_experiment(experiment_name)
mlflow.tensorflow.autolog()


params = {"height": height
    , "width": width
    ,"n_channels": n_channels
    ,"normalisation": ">1000/1000"}

inputs = Input(shape=(height, width, n_channels), name="MSI")
#encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)

def convolution_block(x, filters):
    """
    This function builds the Conv2D block with variable filters

    Arguments:
        x: (Layer) previous layer of type Tensorflow.keras.layers
        filters : (int) number of filters
    Returns:
        tensorflow.keras.layers.Conv2D block

    """
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def MobileUNet(height, width, n_channels, pretrained=True):
    """
    This function builds the model architecture for MobileUNet

    Arguments:
        height : (int) height of image
        width : (int) width of image
        n_channels : (int) number of channels in image
        pretrained : (boolean) whether to use the Imagenet pretrained weights for encoder

    Returns:
        tensorflow.keras.Model

    """
    # Defining the number of filters at each stage of the decoder
    filters_layer = [16, 32, 64, 128]
    inputs = Input(shape=(height, width, n_channels), name="input_image")

    #Encoder
    if pretrained=="True":
        encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    else:
        encoder = MobileNetV2(input_tensor=inputs, weights=None, include_top=False)
    '''
    The following are the selected skip connecion layers from the MobileNetV2 architecture based on feature map resolution
    "input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu",block_13_expand_relu
    '''
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    #Decoder
    #Layer 3
    x = encoder_output
    x = UpSampling2D((2, 2))(x)
    x_skip_3 = encoder.get_layer(skip_connection_names[3]).output
    x = Concatenate()([x, x_skip_3])
    x = convolution_block(x, filters_layer[3])
    #Layer 2
    x = UpSampling2D((2, 2))(x)
    x_skip_2 = encoder.get_layer(skip_connection_names[2]).output
    x = Concatenate()([x, x_skip_2])
    x = convolution_block(x, filters_layer[2])
    #Layer 1
    x = UpSampling2D((2, 2))(x)
    x_skip_1 = encoder.get_layer(skip_connection_names[1]).output
    x = Concatenate()([x, x_skip_1])
    x = convolution_block(x, filters_layer[1])
    #Layer 0
    x = UpSampling2D((2, 2))(x)
    x_skip_0 = encoder.get_layer(skip_connection_names[0]).output
    x = Concatenate()([x, x_skip_0])
    x = convolution_block(x, filters_layer[0])
    #Output Layer
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    model = Model(inputs, x)
    return model

model = MobileUNet(height, width, n_channels, pretrained=True)
model.summary()


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
        threshold : (float) threshold over which pixel is considered positive

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

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

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

metrics = ["acc",
          # tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
          # tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
           iou]
model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-2),  # this LR is overriden by base cycle LR if CyclicLR callback used
    # loss=dice_coef_loss,
    # metrics=dice_score,
    # loss="binary_crossentropy",
    # metrics=metrics
    loss=jaccard_coef_loss,
    metrics=['binary_crossentropy', jaccard_coef_int]
)

# model.fit_generator(
#     batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
#     nb_epoch=nb_epoch,
#     verbose=1,
#     samples_per_epoch=batch_size * 400,
#     callbacks=callbacks,
#     nb_worker=8
#     )


# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.000001, verbose=1, mode='min')

with mlflow.start_run() as run:
    mlflow.log_params(params)
    history = model.fit(train_generator,
    steps_per_epoch=250//6,shuffle=True,
    epochs=70,
    verbose=1,
    validation_data = test_generator,callbacks=[
#            early_stopping,
            reduce_lr
                                                ])
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



val_generator = DataGenerator_segmentation(test, dimension=(256, 256),
                 n_channels=4)
x=val_generator.__getitem__(1)
print('Running predictions...')
predictions = model.predict(val_generator, verbose=1)
print(predictions[0])

plt.imshow(predictions[0])
plt.show()
