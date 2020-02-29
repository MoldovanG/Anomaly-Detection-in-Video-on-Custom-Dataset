import os

import cv2
import mxnet as mx
import numpy as np

from gluoncv import  utils,data
from training.DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from training.ObjectDetector import ObjectDetector
from matplotlib import pyplot as plt


class ModelEvaluator:

    def __init__(self,trainer_stage2 : DataSetTrainer_Stage2, dataset_directory_path):
        self.trainer_stage2 = trainer_stage2
        self.dataset_directory_path = dataset_directory_path
        self.object_detector = ObjectDetector()


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
            frame = frames[i]
            frame_d3 = frames[i - 3]
            frame_p3 = frames[i + 3]
            feature_vectors, bounding_boxes = self.__get_feature_vectors_and_bboxes(frame, frame_d3, frame_p3)
            print('\r', 'Number of frames processed : %d ..... ' % (i), end='', flush=True)
            x, img = data.transforms.presets.ssd.transform_test(frame, short=512)
            printable_frame = img
            for idx,vector in enumerate(feature_vectors):
                score = self.trainer_stage2.get_inference_score(vector)
                c1,l1,c2,l2 = bounding_boxes[idx]
                if score == 0:
                    top_corner = (c1,l1)
                    bottom_corner = (c2,l2)
                    print(top_corner," ; ",bottom_corner," score :: ",score)
                    cv2.rectangle(printable_frame,top_corner,bottom_corner,color=(0,0,0), thickness=2)
            cv2.imshow("frame", printable_frame)
            cv2.waitKey(0)
            # ax = utils.viz.plot_bbox(printable_frame, bounding_boxes, thresh=-100)
            # plt.show()

    def __get_feature_vectors_and_bboxes(self, frame, frame_d3, frame_p3):
      """
      Given 3 frames that represent a certain frame at time t, t-3,and t+3, returns a list of feature_vectors obtained
      by running all the detected objects and their respective gradients through the pretrained autoencoders, and then concatenate
      the result in order to obtain the feature vector.

      :param frame: mxnet.NDarray - the frame that need to be analysed.
      :param frame_d3 : mxnet.NDarray - the frame corresponding to t-3 compared to the initial frame. d3 comes from delta3
      :param frame_p3 : mxnet.NDarray  - the frame corresponding to t+3 compared to the initial frame. p3 comes from plus3
      """
      object_detector = ObjectDetector()
      bounding_boxes,score = object_detector.get_bboxes_and_scores(frame)
      new_score = score[0].asnumpy()
      bboxes = bounding_boxes[0].asnumpy()
      bboxes = self.filter_bboxes(bboxes,new_score)

      cropped_detections, cropped_d3, cropped_p3  = object_detector.get_detections_and_cropped_sections(frame,frame_d3,frame_p3,bounding_boxes,score)
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
      return np.array(list_of_feature_vectors),bboxes


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