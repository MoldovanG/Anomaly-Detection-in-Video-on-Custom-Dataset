import cv2
import numpy as np

from utils.GradientCalculator import GradientCalculator
from utils.ObjectDetector import ObjectDetector


class VideoProcessor:
    """
    Class used for reading and processing a video
    The scope is to obtain all the detected objects in all the frames toghether with their computed gradient.

    Attributes
    ----------
    detected_objects - np.array
    object_gradients - np.array containing the corresponding gradients

    """

    def __init__(self, video_path):
        self.__video_path = video_path
        self.__video = cv2.VideoCapture(video_path)
        self.__detected_objects, self.__object_gradients = self.__process_video(self.__video)

    def __process_video(self, video):
        gradients = []
        objects = []
        counter = 1
        print("Processing video starts ...")
        while True:
            ret, frame = video.read()
            if ret == 0:
                break
            frame = cv2.resize(frame, (640, 640))
            object_detector = ObjectDetector(frame)
            detections = object_detector.get_object_detections()
            for image in detections:
                gradient_calculator = GradientCalculator()
                sobel = gradient_calculator.calculate_gradient(image)
                objects.append(np.resize(image, (64, 64, 1)))
                gradients.append(np.resize(sobel, (64, 64, 1)))
            print('\r',' Number of frames processed : %d ..... ' % (counter), end='',flush='True')
            counter = counter + 1
        print()

        return np.array(objects), np.array(gradients)

    def get_detected_objects(self):
        return self.__detected_objects

    def get_object_gradients(self):
        return self.__object_gradients

    def get_video(self):
        return self.__video


