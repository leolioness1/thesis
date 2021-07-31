import rasterio
import glob
import helper as h
import os
import numpy as np
import matplotlib.pyplot as plt
final = sorted(glob.glob(r'C:\Users\leo__\PycharmProjects\Perma_Thesis\clipped\*.tiff'))
final_original = sorted(glob.glob(r'C:\Users\leo__\PycharmProjects\Perma_Thesis\MSI\thaw\*.tif'))

# window = 64
# # Create training arrays
# data = []
# for img in final:
#     with rasterio.open(img, 'r') as src:
#
#         name = os.path.splitext(os.path.basename(img))[0]
#         print(name)
#         out_meta = src.meta.copy()
#         out_meta.update({"driver": "GTiff",
#                          "height": window,
#                          "width": window,
#                          "count": 8})
#         data.extend(h.create_training_arrays(src,window=window, name=name,out_meta= out_meta))
# #         print(src.read().shape)
original_data=[]
original_data_blank_list=[]
for img in final_original:
    with rasterio.open(img, 'r') as src:
        data_temp=src.read()
        np.nan_to_num(data_temp, nan=0, copy=False)
        if (data_temp[:4]!=0).sum()==0:
            name = os.path.splitext(os.path.basename(img))[0]
            print(f'{name}:possible blank :(')
            original_data_blank_list.append(name)
        else:
            original_data.append(src)

print(len(data))
data[0].shape

neg = 0
pos = 0

for d in data:
    pos_sum = (d[0] == 1).sum()
    if pos_sum == 0:
        neg += 1
    else:
        pos += 1

for d in data:
    pos_sum = (d[3:7]!=0).sum()
    if pos_sum == 0:
        neg += 1
    else:
        print(pos_sum)
        pos += 1



print('Negative: ', neg)
print('Positive: ', pos)
data_pos = [d for d in data if (d[0] == 1).sum() != 0]
len(data_pos)

data_pos[0].shape


example_path= r'/output_windows/T04WFA_20190611_10m_all_Labelled_site_1_clipped_window_58.tiff'
def load_mask(image_path):
    img_object = rasterio.open(image_path)
    img=img_object.read()
    #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
    mask = img[0,:64,:64]
    mask=mask/1000
    mask_final = np.moveaxis(mask, 0, -1)
    np.nan_to_num(mask_final, nan=0,copy=False)#Change nans from data to 0 for mask
    return mask_final

load_mask(example_path).shape
plt.imshow(load_mask(example_path))



from glob import glob
import rasterio
import rasterio.mask

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

VAL_LOCAL_PATH= '../Perma_Thesis/RGB-thawslump-UTM-Images/batagay/'


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

print("Train")
print(train.count())
print("val")
print(val.count())
print("test")
print(test.count())

#Custom data generator that replaces PIL function on image read of tiff record with rasterio
#as default Imagedatagenertator seems to be throwing error

from data_generator_segmentation_adam_data import DataGenerator_segmentation
import dill

height = 64
width = 64
n_channels =4
train_generator = DataGenerator_segmentation(train,dimension=(height,width),batch_size=3, n_channels=n_channels)
val_generator = DataGenerator_segmentation(val,dimension=(height,width),batch_size=3, n_channels=n_channels)
test_generator = DataGenerator_segmentation(test,dimension=(height,width),batch_size=3, n_channels=n_channels)

#Add code for resize
#Add code for normalize range to 0-1
#Add code fro augmentations

x_ex=train_generator.__getitem__(0)

# for i in range(0, len(predictions)):
#     print(np.max(predictions[i]))
#     list.append(np.max(predictions[i]))
#     plot_sample(i)
#
# print(list)
plt.imshow(x_ex[1][6])
(x_ex[0][6][1]).sum() == 0

def plot_sample(ix=None):
    """Function to plot the results"""
    fig, ax = plt.subplots(20, 2, figsize=(10, 10)) #4
    for ix in range(0, 20):

        ax[ix][0].imshow(x_ex[0][ix])

        ax[ix][0].contour(x_ex[1][ix].squeeze(), colors='w', levels=[0.5])
        # ax[ix][0].set_title('Image')

        ax[ix][1].imshow(x_ex[1][ix].squeeze())
        # ax[ix][1].set_title('Mask')

        # ax[ix][2].imshow(predictions[ix].squeeze(), vmin=0, vmax=1)
        #
        # ax[ix][2].contour(test_gt[1][ix].squeeze(), colors='w', levels=[0.5])
        # ax[ix][2].set_title('Predicted')


plot_sample()