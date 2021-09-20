#
# import os
# import random
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
# from tqdm import tqdm_notebook, tnrange
# from itertools import chain
# from skimage.io import imread, imshow, concatenate_images
# from skimage.transform import resize
# from skimage.morphology import label
# from sklearn.model_selection import train_test_split
#
# import tensorflow as tf
#
# from keras.models import Model, load_model
# from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
# from keras.layers.core import Lambda, RepeatVector, Reshape
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
# from keras.layers.merge import concatenate, add
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# import rasterio
# from glob import glob
# # Set some parameters
# im_width = 64
# im_height = 64
# LOCAL_PATH= '../Perma_Thesis/MSI/thaw'
#
# df_dataset = pd.DataFrame()
# files = glob(LOCAL_PATH + "**/*.tif")
# df_dataset['image_paths'] = files
#
#
# def load_image(image_path):
#       """Load grayscale image
#       :param image_path: path to image to load
#       :return: loaded image
#       """
#       img_object = rasterio.open(image_path)
#       img=img_object.read()
#       #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
#       channels=4
#       size=64
#       img_temp = img[:channels,:256,:256]
#       img_temp[img_temp > 1000] = 1000
#       img_temp = img_temp / 1000
#       img_final = np.moveaxis(img_temp, 0, -1)
#         #Reducing image size to 40*40 crop from centre based on finding on 12/03 on thaw slump size being avg
#         #400m so 40 pixels
#       startx = 128 #(128-size/2)
#       starty = 128 #(128-size/2)
#       img_final = img_final[startx:startx+size,startx:starty+size,:]
#       return img_final
# def load_mask(image_path):
#     img_object = rasterio.open(image_path)
#     img=img_object.read()
#     #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
#     size = 128
#     mask = img[-1,:256,:256]
#     mask_final = np.moveaxis(mask, 0, -1)
#     startx = 64  # (128-size/2)
#     starty = 64  # (128-size/2)
#     mask_final = mask_final[startx:startx + size, startx:starty + size]
#     np.nan_to_num(mask_final, nan=0,copy=False)#Change nans from data to 0 for mask
#     return mask_final
#
# # Visualize any randome image along with the mask
# ix = random.randint(0, len(df_dataset['image_paths'].to_list()))
#
# path_list=df_dataset['image_paths'].to_list()
# def plot_sample(ix=None):
#     """Function to plot the results"""
#     if ix is None:
#         ix = random.randint(0, len(path_list)
#                             )
#     has_mask = load_mask(path_list[ix]).max() > 0
#
#     fig, ax = plt.subplots(1, 2, figsize=(20, 10)) #4
#     ax[0].imshow(load_image(path_list[ix]))
#     if has_mask:
#         ax[0].contour(load_mask(path_list[ix]).squeeze(), colors='w', levels=[0.5])
#     ax[0].set_title('Image')
#
#     ax[1].imshow(load_mask(path_list[ix]).squeeze())
#     ax[1].set_title('Mask')
#
#     ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
#     if has_mask:
#         ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[2].set_title('Predicted')
#     #
#     # ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
#     # if has_mask:
#     #     ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     # ax[3].set_title('Salt Predicted binary');
# plot_sample()




import pandas as pd
import matplotlib.pyplot as plt


def pre_read_history_csv(path,param,n=1):
    df = pd.read_csv(path)
    score_zip = dict(zip(df.iloc[-2,0:5].to_list(),df.iloc[-1,0:5].astype(float).round(decimals=2).to_list()))
    df = df.iloc[:-2,1:].astype(float).round(decimals=2)
    latex_input = f"{param} & {df.iloc[-n]['val_dice_coef']} & {df.iloc[-n]['dice_coef']} & {score_zip['dice_coef']} & {df.iloc[-n]['val_loss']} & {df.iloc[-n]['loss']} & {score_zip['loss']}  \\"
    return latex_input

def read_history_csv(path):
    df= pd.read_csv(path)
    df= df.iloc[:-2,1:].astype(float).round(decimals=4)
    return df


def plot_metric(metric, metric_name,parameter_name, df_list, legend_list, n=50):
    # train_metrics = df[metric]
    marker_list=['o','s','*','x','d','h','o','s','*','x','d','h','o','s','*','x','d','h','*','x','d','h','o','s','*','x','d','h']
    for i, df in enumerate(df_list):
        Dynamic_Variable_Name = f'val_metrics_{i}'
        vars()[Dynamic_Variable_Name] = df['val_' + metric]
        # val_metrics_1 = df1['val_' + metric]
        # val_metrics_2 = df2['val_' + metric]
        # val_metrics_3 = df3['val_' + metric]
        # val_metrics_4 = df4['val_' + metric]
        epochs = range(0, len(df))
        plt.plot(epochs, vars()[Dynamic_Variable_Name], marker=marker_list[i])
    # plt.plot(epochs, val_metrics_2, marker='+')
    # plt.plot(epochs, val_metrics_3,marker='>')
    # plt.plot(epochs, val_metrics_4, marker='>')
    plt.title(f'Model comparison of {parameter_name}')
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.ylabel(metric_name)
    plt.legend(legend_list)
    # plt.legend(legend_list,loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = len(legend_list)//2)
    plt.show()
    plt.savefig(fr'C:\Users\leo__\Downloads\thesis_images\experiment plots new\{parameter_name}_{metric_name}.png')
    plt.close()

# parameter_name='patch size'
# param_list_var= ['32','64','128','256']
# parameter_name='normalisation method'
# param_list_var=['max','naive','z_score']
# parameter_name='augmentation method'
# param_list_var=['both','rotation','translation']
# parameter_name='loss function'
# param_list_var=['crossentropy_dice_loss','ce_jaccard_loss','iou_loss','binary_focal_loss','dice_coef_loss']
# parameter_name='optimiser sgd'
# param_list_var= ['adam','nadam','sgd','rmsprop']
# parameter_name='learning rate '
# param_list_var= [0.01,0.001,0.0001]
#parameter_name='batch size'
# # param_list_var=[1,2,3,4,5,6,10,15,25,30]#
#param_list_var=[1,4,6,10,20]
#parameter_name='activation function'
#param_list_var=['elu','relu','tanh']
# parameter_name='initilisation method'
# param_list_var=['he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform', 'random_normal', 'random_uniform']
# parameter_name='loss function selu'
# param_list_var=['dice_coef_loss','crossentropy_dice_loss','iou_loss','ce_jaccard_loss','binary_focal_loss']
parameter_name='planetary'
param_list_var=['']
param_list_var_2=['']
def plot_and_table(parameter,param_list_1,param_list_2,n=200):
    list_df=[]
    list_legend=[]
    table_content=[]

    for i in param_list_1 :
        #CHANGE PATH
        #norm
        # path = f'history_files_norm_selection\history_{i}_4_crossentropy_dice_loss_adam_0.0001_64_50.csv'
        # path= f'history_files_experiment_fixed_normalisation/history_{i}_4_crossentropy_dice_loss_adam_0.0001_64_elu_he_normal_50.csv'
        # path = f'history_files_adam_new_data_recontruct_naive_1/history_{i}_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200from_scratch_es.csv'
        #patch_size
        #path = f'history_files_patch_selection/history_naive_4_crossentropy_dice_loss_adam_0.0001_{i}_50.csv'
        #optimiser
        # path= f'history_files_experiment_fixed_optimiser_sgdtoo/history_naive_4_crossentropy_dice_loss_{i}_0.01_64_elu_he_normal_50.csv'
        #lr
        # path= f'history_files_optimiser_batch_lr_selection/history_z_score_4_crossentropy_dice_loss_rmsprop_{i}_64_50.csv'
        #path=f'history_files_experiment_fixed_optimiser_sgdtoo/history_naive_4_crossentropy_dice_loss_rmsprop_{i}_64_elu_he_normal_50.csv'
        #batch
        #path=f'history_files_optimiser_batch_lr_selection/history_z_score_{i}_crossentropy_dice_loss_rmsprop_0.001_64_50.csv'
        #path=f'history_files_batch_selection_new/history_z_score_{i}_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_normal_100.csv'
        #path = f'history_files_experiment_fixed_batch/history_naive_{i}_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_50.csv'
        #path = f'history_files_experiment_fixed_loss_selu/history_naive_{i}_crossentropy_dice_loss_rmsprop_0.0001_64_selu_he_normal_50.csv'
        #act
        # path=f'history_files_activation_selection/history_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_{i}_50.csv'
        #path=f'history_files_act_selection_new/history_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_{i}_he_uniform_100.csv'
        # path=f'history_files_experiment_fixed_act/history_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_{i}_he_normal_50.csv'
        #path=f'history_files_experiment_fixed_act_2/history_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_{i}_he_normal_50.csv'
        #init
        #path = f'history_files_init_selection\history_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_relu_{i}_50.csv'
        #path=f'history_files_experiment_fixed_init/history_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_{i}_50.csv'
        #loss
        #path=f'history_files_loss_selection_new/history_z_score_6_{i}_rmsprop_0.001_64_elu_he_uniform_100.csv'
        # path=f'history_files_loss_iou_selection_ne/history_z_score_6_{i}_rmsprop_0.001_64_elu_he_uniform_100.csv'
        # path=f'history_files_loss_fixed_selection_new/history_z_score_6_{i}_rmsprop_0.001_64_elu_he_uniform_100.csv'
        # path=f'history_files_experiment_fixed_early_stopping/history_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200{i}.csv'
        #path =f'history_files_experiment_fixed_loss/history_naive_1_{i}_rmsprop_0.0001_64_elu_he_normal_50.csv'
        path=f'history_files_adam_new_data_recontruct_naive_1/history_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200planetretrain.csv'
        #path= f'history_files_experiment_fixed_no_dropout/history_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200.csv'
        list_df.append(read_history_csv(path))
        #list_legend.append(f'{i}')

        table_content.append(pre_read_history_csv(path,i,n=1))
        # list_legend.append(f'{i}')
    for i in param_list_2:
        path = f'history_files_adam_new_data_naive_planetary\history_naive_1_crossentropy_dice_loss_rmsprop_0.0001_64_elu_he_normal_200.csv'
        table_content.append(pre_read_history_csv(path, i, n=1))
        list_df.append(read_history_csv(path))
    list_legend = ['Retraining','Training from scratch']
    plot_metric('dice_coef', 'Validation dice coefficient', parameter, list_df, list_legend)
    # plot_metric('jaccard_coef_int','Validation jaccard coefficient', parameter_name, list_df, list_legend)
    plot_metric('loss', 'Validation loss', parameter, list_df, list_legend)
    with open(fr'C:\Users\leo__\Downloads\thesis_images\table_content\{parameter}.txt', 'w', encoding='utf-8') as f:
        f.write('\ '.join(table_content))
    return ' '.join(table_content)

plot_and_table(parameter_name,param_list_var,param_list_var_2)

# list_df=[]
# list_legend=[]
# table_content=[]
# parameter_name='patch size'
# for i in ['32','64','128','256']:
#     path_name=f'history_files_patch_selection/history_naive_4_crossentropy_dice_loss_adam_0.0001_{i}_50.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(i)
#     table_content.append(pre_read_history_csv(path_name,i))
# print('\ '.join(table_content))
# parameter_name='normalisation method'
# for i in ['max','naive','z_score']:
#     path_name=f'history_files_why_selection/history_{i}_4_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_normal_100.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')

# parameter_name='augmentation method'
# for i in ['both','rotation','translation']:
#     path_name=f'history_files_augmentation/history_z_score_4_crossentropy_dice_loss_adam_0.0001_64_{i}_50.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')

# parameter_name='learning rate'
# for i in [0.001,0.0001]:
#     path_name=f'history_files_opt_selection/history_naive_4_crossentropy_dice_loss_rmsprop_{i}_64_elu_he_normal_100.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')


# parameter_name='batch size'
# for i in [1,2,4,6,10,16,20]:
#     path_name=f'history_files_opt_selection/history_naive_{i}_crossentropy_dice_loss_rmsprop_0.001_64_elu_he_normal_100.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')

# parameter_name='optimiser'
# for i in ['adam','nadam','sgd','rmsprop']:
#     path_name=f'history_files_opt_selection/history_naive_4_crossentropy_dice_loss_{i}_0.001_64_elu_he_normal_100.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')

# parameter_name='loss function'
# #['crossentropy_dice_loss','ce_jaccard_loss','iou_loss','binary_focal_loss','dice_coef_loss']
# for i in ['crossentropy_dice_loss','crossentropy_coef_dice_loss']:
#     #path_name=f'history_files_loss_naive_selection/history_naive_10_{i}_rmsprop_0.001_64_elu_random_uniform_100.csv'
#     #path_name=f'history_files_loss_naive_selection/history_naive_10_{i}_rmsprop_0.001_64_elu_he_normal_100.csv'
#     #path_name=f'history_files_translation_naive_selection/history_naive_10_{i}_rmsprop_0.001_64_elu_he_uniform_100.csv'
#     #path_name=f'history_files_z_score_loss_selection/history_z_score_10_{i}_rmsprop_0.001_64_elu_he_uniform_100.csv'
#     path_name=f'history_files_z_score_ce_loss_selection/history_z_score_10_{i}_rmsprop_0.001_64_elu_he_uniform_100.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')

# parameter_name='activation function'
# for i in ['elu','relu','gelu','selu','tanh']:
#     # path_name=f'history_files_activation_selection/history_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_{i}_50.csv'
#     #path_name = f'history_files_init_act_selection/history_z-score_6_crossentropy_dice_loss_rmsprop_0.001_64_{i}_he_normal_50.csv'
#     #path_name=f'history_files_activation_naive_selection/history_naive_10_crossentropy_dice_loss_rmsprop_0.001_64_{i}_he_normal_100.csv'
#     path_name=f'history_files_z_score_init_selection/history_z_score_10_crossentropy_dice_loss_rmsprop_0.001_64_{i}_he_uniform_100.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')
# parameter_name='initialisation method'
# for i in ['he_normal', 'he_uniform', 'glorot_normal','glorot_uniform','random_normal','random_uniform']:
#     # path_name=f'history_files_init_naive_selection/history_naive_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_{i}_100.csv'
#     #path_name=f'history_files_init_selection\history_z_score_6_crossentropy_dice_loss_rmsprop_0.001_64_relu_{i}_50.csv'
#     path_name = f'history_files_z_score_init_selection/history_z_score_10_crossentropy_dice_loss_rmsprop_0.001_64_elu_{i}_100.csv'
#     list_df.append(read_history_csv(path_name))
#     list_legend.append(f'{i}')
# plot_metric('dice_coef','Validation dice coefficient', parameter_name, list_df, list_legend)
# # plot_metric('jaccard_coef_int','Validation jaccard coefficient', parameter_name, list_df, list_legend)
# plot_metric('loss','Validation loss', parameter_name, list_df, list_legend)
#
# IMAGE_DIR_PATH = '../Perma_Thesis/MSI/thaw/'
# BATCH_SIZE = 4
#
# # create list of PATHS
# image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.tif')]
# from data_generator_segmentation import DataGenerator_segmentation
# val_list=[]
# # val_it=[]
# # Y = np.empty((len(image_paths), *dimension))
# for i, ID in enumerate(image_paths):
#     print(i, ID)
#     image_test,mask_test=load_image(image_path=ID)
#     # Initialization
#
#     # Generate data
#
#         # Store sample
#         # Y_temp = load_mask(ID)
#     # Y[i,] = mask_test
#
#     # mask_test= load_mask(image_path=image)
#
#     # print( "Test Image dimensions:" + str(image_test.shape))
#     #
#     # print( "Test Mask dimensions:" + str(mask_test.shape))
#     val_list.append(tf.math.count_nonzero(mask_test).numpy())
#     # val_it.append(tf.math.count_nonzero(Y[i]).numpy())
#     # plt.imshow(image_test)
#     # plt.show()
#     #
#     # plt.imshow(mask_test)
#     # plt.show()
#
#
# val_list_perc= np.multiply(np.divide(val_list,16384),100)


#
#
# # Create a data frame with the models perfoamnce metrics scores
# models_scores_table = pd.DataFrame({'Logistic Regression': [log['test_accuracy'].mean(),
#                                                             log['test_precision'].mean(),
#                                                             log['test_recall'].mean(),
#                                                             log['test_f1_score'].mean()],
#
#                                     'Support Vector Classifier': [svc['test_accuracy'].mean(),
#                                                                   svc['test_precision'].mean(),
#                                                                   svc['test_recall'].mean(),
#                                                                   svc['test_f1_score'].mean()],
#
#                                     'Decision Tree': [dtr['test_accuracy'].mean(),
#                                                       dtr['test_precision'].mean(),
#                                                       dtr['test_recall'].mean(),
#                                                       dtr['test_f1_score'].mean()],
#
#                                     'Random Forest': [rfc['test_accuracy'].mean(),
#                                                       rfc['test_precision'].mean(),
#                                                       rfc['test_recall'].mean(),
#                                                       rfc['test_f1_score'].mean()],
#
#                                     'Gaussian Naive Bayes': [gnb['test_accuracy'].mean(),
#                                                              gnb['test_precision'].mean(),
#                                                              gnb['test_recall'].mean(),
#                                                              gnb['test_f1_score'].mean()]},
#
#                                    index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
#
# # Add 'Best Score' column
# models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
#
# # Return models performance metrics scores data frame
# return (models_scores_table)