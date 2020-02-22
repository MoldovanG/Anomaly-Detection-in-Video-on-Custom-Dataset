import os

import numpy as np
import cv2
from AutoEncoderModel import AutoEncoderModel
from ObjectDetector import ObjectDetector
from VideoProcessor import VideoProcessor


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
        training_directory = "training_videos_small"
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

    def __get_feature_vectors(self,frame,frame_d3, frame_p3):
        """
        Given 3 frames that represent a certain frame at time t, t-3,and t+3, returns a list of feature_vectors obtained
        by running all the detected objects and their respective gradients through the pretrained autoencoders, and then concatenate
        the result in order to obtain the feature vector.

        :param frame: np.array - the frame that need to be analysed.
        :param frame_d3 : np.array  - the frame corresponding to t-3 compared to the initial frame. d3 comes from delta3
        :param frame_p3 : np.array  - the frame corresponding to t+3 compared to the initial frame. p3 comes from plus3
        """
        object_detector = ObjectDetector()
        cropped_detections, cropped_d3, cropped_p3  = object_detector.get_detections_and_cropped_sections(frame,frame_d3,frame_p3)
        gradients_d3 = self.__get_gradients(cropped_d3)
        gradients_p3 = self.__get_gradients(cropped_p3)
        list_of_feature_vectors = []
        for i in range(cropped_detections.shape[0]):
            apperance_features = self.autoencoder_images.get_encoded_state(np.resize(cropped_detections[i],(64,64,1)))
            motion_features_d3 = self.autoecoder_gradients.get_encoded_state(np.resize(gradients_d3[i],(64,64,1)))
            motion_features_p3 = self.autoecoder_gradients.get_encoded_state(np.resize(gradients_p3[i],(64,64,1)))
            feature_vector = np.concatenate(apperance_features.flatten(),motion_features_d3.flatten(),motion_features_p3.flatten())
            list_of_feature_vectors.append(feature_vector)
        return np.array(list_of_feature_vectors)

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

    def __get_gradients(self, array):
        transformed = []
        for image in array:
            gradient = self.__get_gradient(image)
            transformed.append(gradient)
        return np.array(transformed)

    def __get_gradient(self, image):
        # Get x-gradient in "sx"
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        # Get y-gradient in "sy"
        sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        # Get square root of sum of squares
        sobel = np.hypot(sx, sy)
        sobel = sobel.astype(np.float32)
        sobel = cv2.normalize(src=sobel, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)
        return sobel

