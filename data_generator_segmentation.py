from tensorflow.keras.utils import Sequence
import numpy as np
import rasterio


class DataGenerator_segmentation(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, df, augmentations='None',
                 to_fit=True, batch_size=6, dimension=(128, 128),size=128,norm_method='naive',
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
        self.norm_method=norm_method
        self.size = size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augmentations
        self.max_0= 7876
        self.max_1=8104
        self.max_2=8072
        self.max_3=8360
        self.avg_0=1008.6434025871106
        self.avg_1=896.1888743635184
        self.avg_2=703.2873184155352
        self.avg_3=2104.302326433956
        self.std_0=769.7997710220301
        self.std_1=684.8528019547723
        self.std_2=617.5352071816426
        self.std_3=923.6820035853768
        self.avg_0_m = np.full(
            shape=self.dimension,
            fill_value=self.avg_0,
            dtype=np.float32
        )
        self.avg_1_m = np.full(
            shape=self.dimension,
            fill_value=self.avg_1,
            dtype=np.float32
        )
        self.avg_2_m = np.full(
            shape=self.dimension,
            fill_value=self.avg_2,
            dtype=np.float32
        )
        self.avg_3_m = np.full(
            shape=self.dimension,
            fill_value=self.avg_3,
            dtype=np.float32
        )

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
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)
        y = self._generate_Y(list_IDs_temp)

        if self.to_fit:
            # y = labels_temp
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
    #     Y = np.empty((self.batch_size, 1))
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
        img_temp = img[:channels,:256,:256]
        img_final = np.moveaxis(img_temp, 0, -1)
        #Reducing image size to 40*40 crop from centre based on finding on 12/03 on thaw slump size being avg
        #400m so 40 pixels
        startx = int(128-self.size/2) #(128-size/2)
        starty = int(128-self.size/2)#(128-size/2)
        img_final = img_final[startx:startx+self.size,startx:starty+self.size,:]
        if self.norm_method=='max':
            img_final[:self.size,:self.size,0] = img_final[:self.size,:self.size,0]/self.max_0
            img_final[:self.size,:self.size,1] = img_final[:self.size,:self.size,1] /self.max_1
            img_final[:self.size,:self.size,2] = img_final[:self.size,:self.size,2] /self.max_2
            img_final[:self.size,:self.size,3] = img_final[:self.size,:self.size,3] /self.max_3
        elif self.norm_method=='naive':
            img_final[img_final > 10000] = 10000
            img_final = img_final / 10000
        elif self.norm_method == 'z_score':
            img_final[:self.size,:self.size,0] = (np.subtract(img_final[:self.size,:self.size,0],self.avg_0_m)) /self.std_0
            img_final[:self.size,:self.size,1] = (np.subtract(img_final[:self.size,:self.size,1],self.avg_1_m)) /self.std_1
            img_final[:self.size,:self.size,2] = (np.subtract(img_final[:self.size,:self.size,2],self.avg_2_m)) /self.std_2
            img_final[:self.size,:self.size,3] = (np.subtract(img_final[:self.size,:self.size,3],self.avg_3_m)) /self.std_3
        np.nan_to_num(img_final, nan=0, copy=False)
        return img_final

    #Combine later with image read, as of now inefficient as 2 reads for each file
    def _load_mask(self, image_path):
        img_object = rasterio.open(image_path)
        img=img_object.read()
        #Selecting only 3 channels and fixing size to 256 not correct way exactly but hack
        mask = img[-1,:256,:256]
        mask_final = np.moveaxis(mask, 0, -1)
        np.nan_to_num(mask_final, nan=0,copy=False)#Change nans from data to 0 for mask
        #Reducing image size to 40*40 crop from centre based on finding on 12/03 on thaw slump size being avg
        #400m so 40 pixels
        startx = int(128-self.size/2) #(128-size/2)
        starty = int(128-self.size/2) #(128-size/2)
        mask_final = mask_final[startx:startx+self.size,startx:starty+self.size]
        return mask_final


