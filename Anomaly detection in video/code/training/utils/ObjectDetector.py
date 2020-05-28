
import numpy as np
import cv2
import cvlib as cv
from matplotlib import pyplot as plt
import time

class ObjectDetector:
    """
    Class used for detecting objects inside a given image

     Parameters
    ----------
    image = np.array - the image for which we want to extract the detections

    Attributes
    ----------
    net : pretrained-model from cvlib, using yolov3-tiny architecture trained on the coco dataset.
    threshold : int - the threshold for the detections to be considered positive.
    """
    def __init__(self,image):
        self.image = image
        self.threshold = 0.65
        self.bounding_boxes, self.class_IDs, self.scores= cv.detect_common_objects(image,confidence = self.threshold, model = 'yolov3')
    def __get_cropped_detections(self,frame):
        cropped_images = []
        for idx,score in enumerate(self.scores):
            try:
                c1, l1, c2, l2 = self.bounding_boxes[idx]
                image = frame[l1:l2, c1:c2]
                image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cropped_images.append(image)
            except cv2.error as e:
                print('Possible invalid bounding box :')
                print(l1, l2, c1, c2)
                print('Invalid detection! Shape of the invalid image :')
                print(image.shape)

        return np.array(cropped_images)

    def get_object_detections(self):
        """
        Method used for cropping the detected objects from the given image and returning the images
        reshaped to (64x64) and converted to grayscale for further processing by the autoencoder.

        Returns
        ----
        np.array of size (NxWixHix1) where :
        N = number of detections.
        Wi = 64
        Hi = 64
        """
        detections = self.__get_cropped_detections(self.image)
        return detections


    def get_detections_and_cropped_sections(self,frame_d3,frame_p3):
        """
        Method that will return the detections for the image allready present in the ObjectDetector, and using the
        existent bounding-boxes, will also cropp the frames given as parameters.
        :param frame_d3: np.array
        :param frame_p3: np.array
        :return: A pair formed of :
                        - np.array containg detected object appearence of the t frame.
                        - np.array containg cropped image of the t-3 frame of the corresponding detected object
                        - np.array containg cropped image of the t+3 frame of the corresponding detected object
        """
        detections = self.__get_cropped_detections(self.image)
        cropped_d3 = self.__get_cropped_detections(frame_d3)
        cropped_p3 = self.__get_cropped_detections(frame_p3)

        return detections,cropped_d3,cropped_p3



