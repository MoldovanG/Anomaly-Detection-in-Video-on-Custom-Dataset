import os

import cv2
import mxnet as mx
import numpy as np
import scipy

from gluoncv import utils,data
from training.DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from training.ObjectDetector import ObjectDetector
from matplotlib import pyplot as plt

class ModelEvaluator:

    def __init__(self,trainer_stage2 : DataSetTrainer_Stage2, dataset_directory_path,ground_truth_directory):
        self.trainer_stage2 = trainer_stage2
        self.dataset_directory_path = dataset_directory_path
        self.ground_truth_directory = ground_truth_directory
        self.true_positives = []
        self.false_positives = []
        self.num_gt_detections = 0

    def evaluate_dataset(self):
        for video in os.listdir(self.dataset_directory_path):
            video_number = int(video.split(".")[0])
            video_path = os.path.join(self.dataset_directory_path,video)
            video = cv2.VideoCapture(video_path)
            self.__evaluate_video(video, video_number)


    def __evaluate_video(self, video, video_number):
        frames = []
        counter = 1
        print("Processing video starts ...")
        ground_truth_detections = scipy.io.loadmat(os.path.join(self.ground_truth_directory,str(video_number)+"_label.mat")).get('volLabel')
        while True:
            ret, frame = video.read()
            if ret == 0:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = mx.nd.array(frame)
            frame = frame.astype(np.uint8)
            frames.append(frame)
            counter = counter + 1

        for i in range(3, len(frames) - 3):
            frame_ground_truth = ground_truth_detections[0][i]
            self.num_gt_detections = self.num_gt_detections + self.__count_detection_boxes(frame_ground_truth)
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
                c1 = int(c1/ratio2)-1
                c2 = int(c2/ratio2)-1
                l1 = int(l1/ratio1)-1
                l2 = int(l2/ratio2)-1
                if score == 0:
                    top_corner = (c1,l1)
                    bottom_corner = (c2,l2)
                    print(top_corner," ; ",bottom_corner," score :: ",score)
                    if self.__evaluate_detection(frame_ground_truth,(c1,l1,c2,l2)) is True:
                        self.true_positives.append(1)
                        self.false_positives.append(0)
                        cv2.rectangle(copy_frame, top_corner, bottom_corner, color=(0, 255, 0), thickness=2)
                    else:
                        self.true_positives.append(0)
                        self.false_positives.append(1)
                        cv2.rectangle(copy_frame, top_corner, bottom_corner, color=(255, 0, 0), thickness=2)
            # cv2.imshow("frame", copy_frame)
            # cv2.waitKey(0)


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

    def __compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

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

    def __count_detection_boxes(self, frame_ground_truth):
        counter = 0
        copy_frame = np.copy(frame_ground_truth)
        for i in range(copy_frame.shape[0]):
            for j in range(copy_frame.shape[1]):
                if copy_frame[i][j] == 1:
                    counter = counter + 1
                    self.__fill(copy_frame, i, j, 0)
        return counter

    def __fill(self, copy_frame, i, j, value):
        dl = [-1,1,0,0]
        dc = [0,0,1,-1]
        l= i
        c= j
        while True:
            ok = 0
            old_value = copy_frame[l][c]
            copy_frame[l][c] = value
            for k in range(len(dl)):
                di = l + dl[k]
                dj = c + dc[k]
                if 0 <= di < copy_frame.shape[0] and 0 <= dj < copy_frame.shape[1]:
                    if copy_frame[di][dj] == old_value:
                        l = di
                        c = dj
                        ok = 1
                        break
            if ok == 0:
                break;
    def show_average_precision(self):
        cum_false_positive = np.cumsum(np.array(self.false_positives))
        cum_true_positive = np.cumsum(np.array(self.true_positives))
        rec = cum_true_positive / self.num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.__compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join("/home/george/Licenta/Anomaly detection in video/pictures", 'precizie_medie.png'))
        plt.show()
        print("Accuraccy is :",
              str(max(cum_true_positive) * (100 / (max(cum_true_positive) + max(cum_false_positive)))), "%")
        print("Num of gt_detections:", str(self.num_gt_detections))



