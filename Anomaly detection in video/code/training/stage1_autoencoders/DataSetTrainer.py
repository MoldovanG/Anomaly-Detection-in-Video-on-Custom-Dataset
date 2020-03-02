import os

import numpy as np
import cv2
from code.training.utils.AutoEncoderModel import AutoEncoderModel
from code.training.stage1_autoencoders.VideoProcessor import VideoProcessor

class DataSetTrainer:
    """
    Class used for wrapping up the training process on a specific dataset.
    Path of the directory that contains the training videos is needed in order to create a new instance.
    During initialization, the DataSetTrainer will proceed with full-stack training on the data,
    and at the end the inference phase will be available for the user to check test data.
    """

    def __init__(self, dataset_directory_path):
        """
        :param dataset_directory_path: path of the dataset directory. This directory has to containg the following folders:
            - training_videos : must be non empty and needs to contain all the videos neede for training.
            - detected_objects
            - detected_gradients
        """
        self.dataset_directory_path = dataset_directory_path
        self.total_objects, self.total_gradients = self.__get_objects_and_gradients(verbose=1)
        self.total_objects= self.prepare_data_for_CNN(self.total_objects)
        self.total_gradients = self.prepare_data_for_CNN(self.total_gradients)
        self.autoencoder_images = AutoEncoderModel(self.total_objects, 'raw_object_autoencoder')
        self.autoecoder_gradients = AutoEncoderModel(self.total_gradients, 'gradient_object_autoencoder')

    def __get_objects_and_gradients(self, verbose=0):
        """
        Method used for extracting all the detected objects and their gradients from a given dataset for
        further use in training the autoencoders.

        :param verbose:  Set to 1 for extra prints.
        :return: the detected object and gradients as np.arrays
        """
        gradients = np.resize([], (0, 64, 64, 1))
        objects = np.resize([], (0, 64, 64, 1))
        training_directory = "training_videos"
        for video_name in os.listdir(os.path.join(self.dataset_directory_path, training_directory)):
            print(video_name)
            video_name_without_extension = video_name.split(".")[0]
            video_objects_save_point = os.path.join(self.dataset_directory_path, "detected_objects",
                                                    video_name_without_extension + "_objects")
            video_gradients_save_point = os.path.join(self.dataset_directory_path, "detected_gradients",
                                                      video_name_without_extension + "_gradients")
            if not os.path.exists(video_objects_save_point) \
                and not os.path.exists(video_gradients_save_point):
                os.makedirs(video_objects_save_point)
                os.makedirs(video_gradients_save_point)
                video_path = os.path.join(self.dataset_directory_path, training_directory, video_name)
                video_processor = VideoProcessor(video_path)
                detected_objects = video_processor.get_detected_objects()
                object_gradients = video_processor.get_object_gradients()
                self.write_to_folder(detected_objects, video_objects_save_point)
                self.write_to_folder(object_gradients, video_gradients_save_point)
            else:
                detected_objects = self.load_from_folder(video_objects_save_point)
                object_gradients = self.load_from_folder(video_gradients_save_point)

            gradients = np.concatenate((gradients, object_gradients))
            objects = np.concatenate((objects, detected_objects))
            if (verbose == 1):
                print('Number of detected objects in video:')
                print(detected_objects.shape)
                print('Number of detected gradients in video:')
                print(object_gradients.shape)
                print('Total number of objects untill now :')
                print(objects.shape)
        return objects, gradients

    def write_to_folder(self, array, folder):
        for num, image in enumerate(array, start=0):
            cv2.imwrite(os.path.join(folder, str(num) + ".bmp"), image[:, :, 0])

    def load_from_folder(self, path):
        array = []
        for img_path in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_path), cv2.IMREAD_GRAYSCALE)
            array.append(np.resize(img, (64, 64, 1)))
            print('\r','Number of images read:',len(array),end='')
        print()

        return np.array(array)

    def prepare_data_for_CNN(self,array):
        transformed = []
        for i in range(array.shape[0]):
            transformed.append(array[i] / 255)
        return np.array(transformed)

