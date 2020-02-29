import os

import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


class AutoEncoderModel:
    """
    Class used for creating autoencoders with the needed architecture .

    Attributes
    ----------
    autoencoder = the full autoencoder model, containing both the encoder and the decoder
    encoder = the encoder part of the autoencoder, sharing the weights with the autoencoder
    """

    def __init__(self, input_images, checkpoints_name):
        self.checkpoints_name = checkpoints_name
        self.checkpoint_dir = '/home/george/Licenta/Anomaly detection in video/Avenue Dataset/checkpoints/checkpoints_%s' % self.checkpoints_name
        self.num_epochs = 100
        self.batch_size = 64
        self.autoencoder, self.encoder = self.__generate_autoencoder()
        self.__train_autoencoder(input_images)

    def __generate_autoencoder(self):
        input_img = Input(shape=(64, 64, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), strides=2, padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

        # compiling the models using Adam optimizer and mean squared error as loss
        optimizer = Adam(lr=10 ** -3)
        encoder.compile(optimizer=optimizer, loss='mse')
        autoencoder.compile(optimizer=optimizer, loss='mse')
        print(autoencoder.summary())
        return autoencoder, encoder

    def __train_autoencoder(self, input_images):
        """
        Parameters
        ----------

        input_images = np.array containing the 64x64x1 images used for training the autoencoder
        This images will serve as both training, and target data for the autoencoder.

        """

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            checkpoint_callback = ModelCheckpoint(filepath=self.checkpoint_dir + '/weights.hdf5', verbose=1,
                                                  save_best_only=True)
            early_stopping_monitor = EarlyStopping(patience=4)
            data_train, data_test, gt_train, gt_test = train_test_split(input_images, input_images, test_size=0.20,
                                                                        random_state=42)
            self.autoencoder.fit(data_train, data_train,
                                 epochs=self.num_epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(data_test, data_test),
                                 callbacks=[checkpoint_callback, early_stopping_monitor])
        else:
            self.autoencoder.load_weights(self.checkpoint_dir + '/weights.hdf5')

    def get_encoded_state(self, image):
        """
        Parameters
        ----------
        images - np.array containing the image that need to be encoded

        Returns
        -------
        np.array containing the encoded images, predicted by the encoder.
        """
        input = np.expand_dims(image,axis = 0)
        encodings = self.encoder.predict(input)
        return encodings[0]
