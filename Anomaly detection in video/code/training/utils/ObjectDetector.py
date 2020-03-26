from gluoncv import model_zoo, data, utils
import numpy as np
import cv2
from matplotlib import pyplot as plt

class ObjectDetector:
    """
    Class used for detecting objects inside a given image

     Parameters
    ----------
    image = mxnet.NDarray - the image for which we want to extract the detections

    Attributes
    ----------
    net : pretrained-model from gluoncv, using ssd architecture trained on the coco dataset.
    threshold : int - the threshold for the detections to be considered positive.
    """

    def __init__(self,image):
        self.net = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True)
        self.image = image
        self.threshold = 0.5
        self.x_transformed_image, self.img_transformed_image = data.transforms.presets.ssd.transform_test(image, short=512)
        self.class_IDs, self.scores, self.bounding_boxes = self.net(self.x_transformed_image)
        self.bounding_boxes,self.scores = self.__clean_bounding_boxes_and_scores(self.bounding_boxes[0].asnumpy(), self.scores[0].asnumpy())


    def get_bounding_box_coordinates(self, bounding_boxes, index):
        """
        Parameters
        ----------
        bounding_boxes = np.array(Nx4)list containing all the bounding_boxes coordinates
        index = int - the index of the wanted bounding-box

        Returns
        ----
        A pair of 4 items(c1,l1,c2,l2)describing the top left and bottom right corners of the bounding box
        """
        c1 = int(bounding_boxes[index][0])
        l1 = int(bounding_boxes[index][1])
        c2 = int(bounding_boxes[index][2])
        l2 = int(bounding_boxes[index][3])
        return c1, l1, c2, l2

    def __get_cropped_detections(self,frame):
        cropped_images = []
        for idx,score in enumerate(self.scores):
            try:
                c1, l1, c2, l2 = self.get_bounding_box_coordinates(self.bounding_boxes, idx)
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
        detections = self.__get_cropped_detections(self.img_transformed_image)
        return detections


    def get_detections_and_cropped_sections(self,frame_d3,frame_p3):
        """
        Method that will return the detections for the image allready present in the ObjectDetector, and using the
        existent bounding-boxes, will also cropp the frames given as parameters.
        :param frame_d3: mxnet.NDArray
        :param frame_p3: mxnet.NDArray
        :return: A pair formed of :
                        - np.array containg detected object appearence of the t frame.
                        - np.array containg cropped image of the t-3 frame of the corresponding detected object
                        - np.array containg cropped image of the t+3 frame of the corresponding detected object
        """
        z, img_d3 = data.transforms.presets.ssd.transform_test(frame_d3, short=512)
        v, img_p3 = data.transforms.presets.ssd.transform_test(frame_p3, short=512)
        detections = self.__get_cropped_detections(self.img_transformed_image)
        cropped_d3 = self.__get_cropped_detections(img_d3)
        cropped_p3 = self.__get_cropped_detections(img_p3)

        return detections,cropped_d3,cropped_p3

    def __clean_bounding_boxes_and_scores(self, bounding_boxes, scores):
        """
        Method used for removing the bounding boxes that have a score under the set threshold
        :param bounding_boxes: bounding_boxes np.array with size(Nx4)
        :param scores: scores np.array with size(Nx1)
        :return: Trimmed bounding_boxes as np.array, and trimmed scoresas np.array
        """
        bboxes = []
        new_scores = []
        counter = 0
        while scores[counter] > self.threshold:
            c1, l1, c2, l2 = self.get_bounding_box_coordinates(bounding_boxes, counter)
            if c1 < 0 or c2 > self.img_transformed_image.shape[1] or l1 < 0 or l2 > self.img_transformed_image.shape[0]:
                print("Invalid bounding box:", c1,l1,c2,l2)
                counter = counter + 1
                continue
            bboxes.append(bounding_boxes[counter])
            new_scores.append(scores[counter])
            counter = counter + 1
        return np.array(bboxes),np.array(new_scores)




