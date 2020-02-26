import cv2
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

from ObjectDetector import ObjectDetector


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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = mx.nd.array(frame)
            frame = frame.astype(np.uint8)
            object_detector = ObjectDetector()
            detections = object_detector.get_object_detections(frame)
            for image in detections:
                # Get x-gradient in "sx"
                sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
                # Get y-gradient in "sy"
                sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
                # Get square root of sum of squares
                sobel = np.hypot(sx, sy)
                sobel = sobel.astype(np.float32)
                sobel = cv2.normalize(src=sobel, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)
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


