import os

import cv2
import mxnet as mx
import numpy as np
import scipy

from gluoncv import  utils,data
from training.DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from training.ObjectDetector import ObjectDetector
from matplotlib import pyplot as plt


class ModelEvaluator:

    def __init__(self,trainer_stage2 : DataSetTrainer_Stage2, dataset_directory_path,ground_truth_directory):
        self.trainer_stage2 = trainer_stage2
        self.dataset_directory_path = dataset_directory_path
        self.ground_truth_directory = ground_truth_directory


    def evaluate_dataset(self):
        for video in os.listdir(self.dataset_directory_path):
            if video == "08.avi":
                video_path = os.path.join(self.dataset_directory_path,video)
                video = cv2.VideoCapture(video_path)
                self.evaluate_video(video)



    def evaluate_video(self, video):
        frames = []
        counter = 1
        print("Processing video starts ...")
        ground_truth_detections = scipy.io.loadmat(os.path.join(self.ground_truth_directory,"8_label.mat")).get('volLabel')
        number_of_detections = 0
        number_of_correct_detections = 0
        while True:
            ret, frame = video.read()
            if ret == 0:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = mx.nd.array(frame)
            frame = frame.astype(np.uint8)
            frames.append(frame)
            counter = counter + 1

        total_feature_vectors = []
        for i in range(3, len(frames) - 3):
            frame_ground_truth = ground_truth_detections[0][i]
            frame = frames[i]
            frame_d3 = frames[i - 3]
            frame_p3 = frames[i + 3]
            feature_vectors, bounding_boxes = self.__get_feature_vectors_and_bboxes(frame, frame_d3, frame_p3)
            print('\r', 'Number of frames processed : %d ..... ' % (i), end='', flush=True)
            x, img = data.transforms.presets.ssd.transform_test(frame, short=512)
            printable_frame = img
            ratio1 = printable_frame.shape[0]/frame.shape[0]
            ratio2 = printable_frame.shape[1]/frame.shape[1]
            copy_frame = frame.asnumpy()
            print(copy_frame.shape)
            for idx,vector in enumerate(feature_vectors):
                score = self.trainer_stage2.get_inference_score(vector)
                c1,l1,c2,l2 = bounding_boxes[idx]
                c1 = int(c1/ratio2)
                c2 = int(c2/ratio2)
                l1 = int(l1/ratio1)
                l2 = int(l2/ratio2)
                if score == 0:
                    number_of_detections = number_of_detections + 1
                    top_corner = (c1,l1)
                    bottom_corner = (c2,l2)
                    print(top_corner," ; ",bottom_corner," score :: ",score)
                    if self.__evaluate_detection(frame_ground_truth,(c1,l1,c2,l2)) is True:
                        number_of_correct_detections = number_of_correct_detections + 1
                        cv2.rectangle(copy_frame, top_corner, bottom_corner, color=(0, 255, 0), thickness=2)
                    else:
                        cv2.rectangle(copy_frame, top_corner, bottom_corner, color=(255, 0, 0), thickness=2)
            cv2.imshow("frame", copy_frame)
            cv2.waitKey(0)
        print("Accuraccy is :", str(number_of_correct_detections* (100/number_of_detections)),"%")

    def __get_feature_vectors_and_bboxes(self, frame, frame_d3, frame_p3):
      """
      Given 3 frames that represent a certain frame at time t, t-3,and t+3, returns a list of feature_vectors obtained
      by running all the detected objects and their respective gradients through the pretrained autoencoders, and then concatenate
      the result in order to obtain the feature vector.

      :param frame: mxnet.NDarray - the frame that need to be analysed.
      :param frame_d3 : mxnet.NDarray - the frame corresponding to t-3 compared to the initial frame. d3 comes from delta3
      :param frame_p3 : mxnet.NDarray  - the frame corresponding to t+3 compared to the initial frame. p3 comes from plus3
      """
      object_detector = ObjectDetector(frame)
      bounding_boxes = object_detector.bounding_boxes

      cropped_detections, cropped_d3, cropped_p3  = object_detector.get_detections_and_cropped_sections(frame_d3,frame_p3)
      gradients_d3 = self.__prepare_data_for_CNN(self.__get_gradients(cropped_d3))
      gradients_p3 = self.__prepare_data_for_CNN(self.__get_gradients(cropped_p3))
      cropped_detections = self.__prepare_data_for_CNN(cropped_detections)
      list_of_feature_vectors = []
      for i in range(cropped_detections.shape[0]):
          apperance_features = self.trainer_stage2.autoencoder_images.get_encoded_state(np.resize(cropped_detections[i], (64, 64, 1)))
          motion_features_d3 = self.trainer_stage2.autoencoder_gradients.get_encoded_state(np.resize(gradients_d3[i], (64, 64, 1)))
          motion_features_p3 = self.trainer_stage2.autoencoder_gradients.get_encoded_state(np.resize(gradients_p3[i], (64, 64, 1)))
          feature_vector = np.concatenate((apperance_features.flatten(),motion_features_d3.flatten(),motion_features_p3.flatten()))
          list_of_feature_vectors.append(feature_vector)
      return np.array(list_of_feature_vectors),bounding_boxes


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

    def __prepare_data_for_CNN(self, array):
        transformed = []
        for i in range(array.shape[0]):
            transformed.append(array[i] / 255)
        return np.array(transformed)

    def filter_bboxes(self, bounding_boxes, score):
        bboxes = []
        counter = 0
        while score[counter] > 0.5:
            bboxes.append(bounding_boxes[counter])
            counter= counter + 1
        return np.array(bboxes)

    def __evaluate_detection(self, frame_ground_truth, bbox):
        """
        Return true if a single detected pixel is also marked as ground truth, false otherwise
        :param frame_ground_truth: frame groun truth matrix (0 marked as non detected pixels, detected pixels are marked with 1)
        :param bbox: the detected bbox
        :return: bool
        """
        c1 = int(bbox[0])
        l1 = int(bbox[1])
        c2 = int(bbox[2])
        l2 = int(bbox[3])
        for i in range (l1,l2):
            for j in range(c1,c2):
                if frame_ground_truth[i][j] == 1:
                    return True
        return False

