from tensorflow.keras.utils import Sequence
import numpy as np
import rasterio

class DataGenerator_segmentation(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, df, augmentations='None',
                 to_fit=True, batch_size=6, dimension=(256, 256),
                 n_channels=4, shuffle=True):
        """Initialization
        :param list_IDs: list of all file ids to use in the generator
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dimension: tuple indicating image dimension
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = df['image_paths'].tolist()
        self.labels = df['label'].tolist()
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dimension = dimension
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augmentations

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)
        y = self._generate_Y(list_IDs_temp)

        if self.to_fit:
            #y = labels_temp
            im, mask = [], []
            for x, y in zip(X, y):
                im.append(x)
                mask.append(y)
            return np.array(im), np.array(mask)
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dimension, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_image(ID)

        return X
    # def _generate_Y(self, labels_temp):
    #     """Generates data containing batch_size images
    #     :param list_IDs_temp: list of label ids to load
    #     :return: batch of images
    #     """
    #     # Initialization
    #     Y = np.empty((self.batch_size,1))
    #
    #     # Generate data
    #     for i, ID in enumerate(labels_temp):
    #         # Store sample
    #         Y[i,] = ID
    #
    #     return Y
    
    def _generate_Y(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        Y = np.empty((self.batch_size, *self.dimension, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            Y_temp = self._load_mask(ID)
            Y[i,] = Y_temp[..., np.newaxis]


        return Y

    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img_object = rasterio.open(image_path)
        img=img_object.read()
        #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
        channels=4
        size=64
        img_temp = img[:channels,:256,:256]
        img_temp = img_temp / 1000
        img_final = np.moveaxis(img_temp, 0, -1)
#         #Reducing image size to 40*40 crop from centre based on finding on 12/03 on thaw slump size being avg
#         #400m so 40 pixels
#         startx = 98 #(128-size/2)
#         starty = 98 #(128-size/2)
#         img_final = img_final[startx:startx+size,startx:starty+size,:]        
        return img_final

    #Combine later with image read, as of now inefficient as 2 reads for each file
    def _load_mask(self, image_path):
        img_object = rasterio.open(image_path)
        img=img_object.read()
        #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
        channels=4
        size=64
        mask = img[-1,:256,:256]
        mask_final = np.moveaxis(mask, 0, -1)
        np.nan_to_num(mask_final, nan=0,copy=False)#Change nans from data to 0 for mask
        #Reducing image size to 40*40 crop from centre based on finding on 12/03 on thaw slump size being avg
        #400m so 40 pixels
#         startx = 98 #(128-size/2)
#         starty = 98 #(128-size/2)
#         mask_final = mask_final[startx:startx+size,startx:starty+size]
        return mask_final