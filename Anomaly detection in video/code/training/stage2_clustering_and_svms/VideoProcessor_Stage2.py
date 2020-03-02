import cv2
import mxnet as mx
import numpy as np

from code.training.utils.AutoEncoderModel import AutoEncoderModel
from code.training.utils.GradientCalculator import GradientCalculator
from code.training.utils.ObjectDetector import ObjectDetector


class VideoProcessor_Stage2:
    """
    Class used for reading and processing a video
    The scope is to obtain all the detected objects in all the frames toghether with their computed gradient.

    Attributes
    ----------
    detected_objects - np.array
    object_gradients - np.array containing the corresponding gradients

    """

    def __init__(self, video_path,autoencoder_images : AutoEncoderModel, autoencoder_gradients : AutoEncoderModel):
        self.__video_path = video_path
        self.__autoencoder_images = autoencoder_images
        self.__autoencoder_gradients = autoencoder_gradients
        self.__video = cv2.VideoCapture(video_path)
        self.__feature_vectors = self.__process_video(self.__video)

    def get_feature_vectors(self):
        return self.__feature_vectors

    def __process_video(self, video):
        frames = []
        counter = 1
        print("Processing video starts ...")
        while True:
            ret, frame = video.read()
            if ret == 0:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = mx.nd.array(frame)
            frame = frame.astype(np.uint8)
            frames.append(frame)
            counter = counter + 1

        total_feature_vectors =[]
        for i in range(3,len(frames)-3):
          frame = frames[i]
          frame_d3 = frames[i-3]
          frame_p3 = frames[i+3]
          feature_vectors = self.__generate_feature_vectors(frame, frame_d3, frame_p3)
          print('\r','Number of frames processed : %d ..... ' % (i), end='',flush=True)
          for vector in feature_vectors:
            total_feature_vectors.append(vector)
        print()
        return np.array(total_feature_vectors)

    def __generate_feature_vectors(self, frame, frame_d3, frame_p3):
      """
      Given 3 frames that represent a certain frame at time t, t-3,and t+3, returns a list of feature_vectors obtained
      by running all the detected objects and their respective gradients through the pretrained autoencoders, and then concatenate
      the result in order to obtain the feature vector.

      :param frame: mxnet.NDarray - the frame that need to be analysed.
      :param frame_d3 : mxnet.NDarray - the frame corresponding to t-3 compared to the initial frame. d3 comes from delta3
      :param frame_p3 : mxnet.NDarray  - the frame corresponding to t+3 compared to the initial frame. p3 comes from plus3
      """
      object_detector = ObjectDetector(frame)
      cropped_detections, cropped_d3, cropped_p3 = object_detector.get_detections_and_cropped_sections(frame_d3,
                                                                                                       frame_p3)
      gradients_d3 = self.__prepare_data_for_CNN( self.__get_gradients(cropped_d3))
      gradients_p3 = self.__prepare_data_for_CNN(self.__get_gradients(cropped_p3))
      cropped_detections = self.__prepare_data_for_CNN(cropped_detections)

      list_of_feature_vectors = []
      for i in range(cropped_detections.shape[0]):
          apperance_features = self.__autoencoder_images.get_encoded_state(np.resize(cropped_detections[i], (64, 64, 1)))
          motion_features_d3 = self.__autoencoder_gradients.get_encoded_state(np.resize(gradients_d3[i], (64, 64, 1)))
          motion_features_p3 = self.__autoencoder_gradients.get_encoded_state(np.resize(gradients_p3[i], (64, 64, 1)))
          feature_vector = np.concatenate((apperance_features.flatten(),motion_features_d3.flatten(),motion_features_p3.flatten()))
          list_of_feature_vectors.append(feature_vector)
      return np.array(list_of_feature_vectors)

    def __get_gradients(self, array):
        transformed = []
        for image in array:
            gradient_calculator = GradientCalculator()
            gradient = gradient_calculator.calculate_gradient(image)
            transformed.append(gradient)
        return np.array(transformed)

    def __prepare_data_for_CNN(self, array):
        transformed = []
        for i in range(array.shape[0]):
            transformed.append(array[i] / 255)
        return np.array(transformed)


