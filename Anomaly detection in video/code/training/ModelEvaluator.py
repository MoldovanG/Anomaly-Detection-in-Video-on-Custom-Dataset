import os
import cv2
import numpy as np
from scipy import io
import time

from scipy.ndimage import gaussian_filter
from stage2_clustering_and_svms.DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from utils.GradientCalculator import GradientCalculator
from utils.ObjectDetector import ObjectDetector
from matplotlib import pyplot as plt

from utils.PrecisionCalculator import PrecisionCalculator


class ModelEvaluator:

    def __init__(self,trainer_stage2 : DataSetTrainer_Stage2, dataset_directory_path,ground_truth_directory):
        self.trainer_stage2 = trainer_stage2
        self.dataset_directory_path = dataset_directory_path
        self.testing_videos_path = os.path.join(dataset_directory_path, "testing_videos")
        self.ground_truth_directory = ground_truth_directory
        self.true_positives = []
        self.false_positives = []
        self.frame_scores = []
        self.video_frame_scores = []
        self.video_ground_truth = []
        self.gt_frame_anormalities = []
        self.num_gt_detections = 0
        self.threshold = 0

    def evaluate_dataset(self):
        names = []
        for video in os.listdir(self.testing_videos_path):
            video_number = int(video.split(".")[0])
            names.append(video_number)
            video_path = os.path.join(self.testing_videos_path,video)
            video = cv2.VideoCapture(video_path)
            self.__evaluate_video(video, video_number)
        self.frame_scores = np.array(self.frame_scores)
        self.gt_frame_anormalities = np.array(self.gt_frame_anormalities)
        precisions = []
        for idx, video_frame_scores in enumerate(self.video_frame_scores):
            video_gt_scores = self.video_ground_truth[idx]
            if len(video_frame_scores) !=0 :
                avg_precision = self.__compute_true_positives_and_false_positives(video_frame_scores,video_gt_scores)
                print(names[idx], "has accuracy: ", avg_precision)
                precisions.append(avg_precision)
            else:
                print(names[idx], " doesn t have frame scores")
        print("Precizia media pe datasetul Avenue este : ", np.mean(np.array(precisions)))

    def visual_results_on_dataset(self):
        for video in os.listdir(self.testing_videos_path):
            video_path = os.path.join(self.testing_videos_path, video)
            video = cv2.VideoCapture(video_path)
            self.__visual_analisys_on_video(video, )
    def __visual_analisys_on_video(self, video):
        frames = []
        print("Processing video starts ...")

        while True:
            ret, frame = video.read()
            if ret == 0:
                break
            frames.append(frame)
        for i in range(3, len(frames) - 3):
            frame = frames[i]
            frame_d3 = frames[i - 3]
            frame_p3 = frames[i + 3]
            feature_vectors, bounding_boxes = self.__get_feature_vectors_and_bboxes(frame, frame_d3, frame_p3)
            if feature_vectors.shape[0] > 0:
                feature_vectors = self.trainer_stage2.normalize_data(feature_vectors)
            for idx, vector in enumerate(feature_vectors):
                score = self.trainer_stage2.get_inference_score(vector)
                c1, l1, c2, l2 = bounding_boxes[idx]
                if score > self.threshold:
                    top_corner = (c1, l1)
                    bottom_corner = (c2, l2)
                    print(top_corner, " ; ", bottom_corner, " score :: ", score)
                    cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, str(round(score, 2)), top_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
            end_time = time.time()
            # print("running final inference for all feature vectors took %f seconds " % (end_time - start_time))
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

    def __evaluate_video(self, video, video_number):
        frames = []
        counter = 1
        video_results_directory = os.path.join('/home/george/Licenta/Anomaly detection in video/Avenue Dataset/results',str(video_number))
        print("Processing video starts ...")
        ground_truth_detections = io.loadmat(os.path.join(self.ground_truth_directory,str(video_number)+"_label.mat")).get('volLabel')

        local_frame_scores = []
        local_gt_scores = []
        num_gt_detections = 0
        if os.path.exists(video_results_directory):
            local_frame_scores = np.load(os.path.join(video_results_directory,str(video_number)+'-scores.npy'))
            local_gt_scores =  np.load(os.path.join(video_results_directory,str(video_number)+'-gt.npy'))
            num_gt_detections = np.count_nonzero(local_gt_scores)
        else:
            os.mkdir(video_results_directory)
            while True:
                ret, frame = video.read()
                if ret == 0:
                    break
                frames.append(frame)
                counter = counter + 1
            for i in range(3, len(frames) - 3):
                frame_ground_truth = ground_truth_detections[0][i]
                detection_boxes_counter = self.__count_detection_boxes(frame_ground_truth)
                if detection_boxes_counter > 0:
                    num_gt_detections = num_gt_detections + 1
                    local_gt_scores.append(1)
                else:
                    local_gt_scores.append(0)

                frame_score = -1
                frame = frames[i]
                frame_d3 = frames[i - 3]
                frame_p3 = frames[i + 3]
                feature_vectors, bounding_boxes = self.__get_feature_vectors_and_bboxes(frame, frame_d3, frame_p3)
                if feature_vectors.shape[0] > 0:
                    feature_vectors = self.trainer_stage2.normalize_data(feature_vectors)
                print('\r', 'Number of frames processed : %d ..... ' % (len(local_frame_scores)), end='', flush=True)
                for idx,vector in enumerate(feature_vectors):
                    score = self.trainer_stage2.get_inference_score(vector)
                    c1,l1,c2,l2 = bounding_boxes[idx]
                    if score > self.threshold:
                        if score > frame_score:
                            frame_score = score
                        top_corner = (c1,l1)
                        bottom_corner = (c2,l2)
                        print(top_corner," ; ",bottom_corner," score :: ",score)
                        if self.__evaluate_detection(frame_ground_truth,(c1,l1,c2,l2)) is True:
                            cv2.rectangle(frame, top_corner, bottom_corner, color=(0, 255, 0), thickness=2)
                            cv2.putText(frame, str(round(score, 2)), top_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, top_corner, bottom_corner, color=(255, 0, 0), thickness=2)
                            cv2.putText(frame, str(round(score, 2)), top_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 2)
                if frame_score < self.threshold:
                    frame_score = 0
                end_time = time.time()
                # print("running final inference for all feature vectors took %f seconds " % (end_time - start_time))
                local_frame_scores.append(frame_score)
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
            np.save(os.path.join(video_results_directory, str(video_number) + '-scores.npy'),local_frame_scores)
            np.save(os.path.join(video_results_directory, str(video_number) + '-gt.npy'),local_gt_scores)
        self.video_frame_scores.append(np.array(local_frame_scores))
        self.video_ground_truth.append(np.array(local_gt_scores))
        for score in local_frame_scores:
            self.frame_scores.append(score)
        for score in local_gt_scores:
            self.gt_frame_anormalities.append(score)
        self.num_gt_detections = self.num_gt_detections + num_gt_detections



    def __get_feature_vectors_and_bboxes(self, frame, frame_d3, frame_p3):
      """
      Given 3 frames that represent a certain frame at time t, t-3,and t+3, returns a list of feature_vectors obtained
      by running all the detected objects and their respective gradients through the pretrained autoencoders, and then concatenate
      the result in order to obtain the feature vector.

      :param frame: mxnet.NDarray - the frame that need to be analysed.
      :param frame_d3 : mxnet.NDarray - the frame corresponding to t-3 compared to the initial frame. d3 comes from delta3
      :param frame_p3 : mxnet.NDarray  - the frame corresponding to t+3 compared to the initial frame. p3 comes from plus3
      """
      start_time = time.time()
      object_detector = ObjectDetector(frame)
      bounding_boxes = object_detector.bounding_boxes
      cropped_detections, cropped_d3, cropped_p3 = object_detector.get_detections_and_cropped_sections(frame_d3,frame_p3)

      end_time = time.time()
      # print("running object detection took %f seconds " % (end_time - start_time))
      gradient_calculator = GradientCalculator()
      gradients_d3 = self.__prepare_data_for_CNN(gradient_calculator.calculate_gradient_bulk(cropped_d3))
      gradients_p3 = self.__prepare_data_for_CNN(gradient_calculator.calculate_gradient_bulk(cropped_p3))
      cropped_detections = self.__prepare_data_for_CNN(cropped_detections)
      list_of_feature_vectors = []
      start_time = time.time()
      for i in range(cropped_detections.shape[0]):
          apperance_features = self.trainer_stage2.autoencoder_images.get_encoded_state(np.resize(cropped_detections[i], (64, 64, 1)))
          motion_features_d3 = self.trainer_stage2.autoencoder_gradients.get_encoded_state(np.resize(gradients_d3[i], (64, 64, 1)))
          motion_features_p3 = self.trainer_stage2.autoencoder_gradients.get_encoded_state(np.resize(gradients_p3[i], (64, 64, 1)))
          feature_vector = np.concatenate((motion_features_d3.flatten(),apperance_features.flatten(), motion_features_p3.flatten()))
          list_of_feature_vectors.append(feature_vector)
          # fig, axs = plt.subplots(1, 3)
          # random = randint(0,99999999)
          # axs[0].imshow((self.trainer_stage2.autoencoder_images.autoencoder.predict(np.expand_dims(np.resize(cropped_detections[i], (64, 64, 1)),axis=0))[0][:,:,0])*255,cmap="gray")
          # axs[1].imshow(self.trainer_stage2.autoencoder_gradients.autoencoder.predict(np.expand_dims(np.resize(gradients_d3[i], (64, 64, 1)),axis=0))[0][:,:,0]*255, cmap="gray")
          # axs[2].imshow(self.trainer_stage2.autoencoder_gradients.autoencoder.predict(np.expand_dims(np.resize(gradients_p3[i], (64, 64, 1)),axis=0))[0][:,:,0]*255, cmap="gray")
          # plt.savefig(os.path.join("/home/george/Licenta/Anomaly detection in video/pictures", 'feature_vectors_predicted'+str(random)+'.png'))
          # plt.close(fig)
          # fig, axs = plt.subplots(1, 3)
          # axs[0].imshow(cropped_detections[i]*255, cmap="gray")
          # axs[1].imshow(gradients_d3[i]*255, cmap="gray")
          # axs[2].imshow(gradients_p3[i]*255, cmap="gray")
          # plt.savefig(os.path.join("/home/george/Licenta/Anomaly detection in video/pictures",
          #                          'feature_vectors' + str(random) + '.png'))
          # plt.close(fig)
      end_time = time.time()
      # print("running encoders for all feature objects took %f seconds " % (end_time - start_time))
      return np.array(list_of_feature_vectors),bounding_boxes

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

    def __compute_true_positives_and_false_positives(self, frame_scores, ground_truth_values):
        best_threshold = 0
        best_gaussian_parameter = 0
        best_avg_precision = 0
        num_gt_detections = np.count_nonzero(ground_truth_values)
        frame_auc_threshold = 0
        max_value = np.max(frame_scores)
        for i in range (35):
            if (frame_auc_threshold > max_value):
                break
            frame_scores_copy = self.apply_threshold(frame_scores, frame_auc_threshold)
            for gaussian_parameter in range (5,100,5):
                true_positives = []
                false_positives = []
                frame_scores_smoothened = gaussian_filter(frame_scores_copy, sigma=gaussian_parameter)
                frame_scores_smoothened = (frame_scores_smoothened - np.min(frame_scores_smoothened)) / (np.max(frame_scores_smoothened) - np.min(frame_scores_smoothened))
                for (idx,score) in enumerate(frame_scores_smoothened):
                    if score > 0 and ground_truth_values[idx] > 0:
                        true_positives.append(1)
                        false_positives.append(0)
                    else:
                        if score > 0 and ground_truth_values[idx] == 0:
                            true_positives.append(0)
                            false_positives.append(1)
                avg_precision = 0
                if len(true_positives) >0 :
                    precision_calculator = PrecisionCalculator()
                    avg_precision = precision_calculator.show_average_precision(true_positives, false_positives,
                                                            num_gt_detections)

                if avg_precision > best_avg_precision:
                    best_avg_precision = avg_precision
                    best_threshold = frame_auc_threshold
                    best_gaussian_parameter = gaussian_parameter
                    # print("Found new best precision :" + str(best_avg_precision) + "for threshold  : " + str(
                    #     best_threshold) + "and parameter:" + str(best_gaussian_parameter))

            frame_auc_threshold = frame_auc_threshold + 0.1

        x = np.linspace(0, frame_scores.shape[0], frame_scores.shape[0])
        frame_scores_copy = self.apply_threshold(frame_scores,best_threshold)
        frame_scores_smoothened = gaussian_filter(frame_scores_copy, sigma=best_gaussian_parameter)
        frame_scores_smoothened = (frame_scores_smoothened - np.min(frame_scores_smoothened)) / (np.max(frame_scores_smoothened) - np.min(frame_scores_smoothened))
        fig, ax = plt.subplots()
        ax.plot(x, ground_truth_values, color="red", label="ground truth")
        ax.fill_between(x, ground_truth_values, alpha=0.2)
        ax.plot(x, frame_scores_smoothened, color="blue", label="frame_scores")
        plt.show()
        plt.close(fig)

        return best_avg_precision

    def apply_threshold(self, frame_scores, threshold):
        new_scores = []
        for score in frame_scores:
            if score > threshold:
                new_scores.append(score)
            else:
                new_scores.append(0)
        return np.array(new_scores)







