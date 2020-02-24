from gluoncv import model_zoo, data, utils
import numpy as np
import cv2


class ObjectDetector:
    """
    Class used for detecting objects inside a given image

    Attributes
    ----------
    net : pretrained-model from gluoncv, using ssd architecture trained on the coco dataset.
    threshold : int - the threshold for the detections to be considered positive.
    """

    def __init__(self):
        self.net = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True)
        self.threshold = 0.5

    def get_bounding_box_coordinates(self, bounding_boxes, index):
        """
        Parameters
        ----------
        bounding_boxes = mxnet.NDarray(Nx4)list containing all the bounding_boxes coordinates
        index = int - the index of the wanted bounding-box

        Returns
        ----
        A pair of 4 items(c1,l1,c2,l2)describing the top left and bottom right corners of the bounding box
        """
        flat_bounding_box = bounding_boxes[index].asnumpy()
        c1 = int(flat_bounding_box.item(0))
        l1 = int(flat_bounding_box.item(1))
        c2 = int(flat_bounding_box.item(2))
        l2 = int(flat_bounding_box.item(3))
        return c1, l1, c2, l2

    def get_cropped_detections(self,img,bounding_boxes,scores):
        cropped_images = []
        counter = 0
        while (scores[0][counter] > self.threshold):
            try:
                c1, l1, c2, l2 = self.get_bounding_box_coordinates(bounding_boxes[0], counter)
                image = img[l1:l2, c1:c2]
                image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cropped_images.append(image)
            except cv2.error as e:
                print('Possible invalid bounding box :')
                print(l1, l2, c1, c2)
                print('Invalid detection! Shape of the invalid image :')
                print(image.shape)
            finally:
                counter = counter + 1

        return np.array(cropped_images)

    def get_object_detections(self, image):
        """
        Method used for cropping the detected objects from the given image and returning the images
        reshaped to (64x64) and converted to grayscale for further processing by the autoencoder.

        Parameters
        ----------
        image = mxnet.NDarray - the image for which we want to extract the detections

        Returns
        ----
        np.array of size (NxWixHix1) where :
        N = number of detections.
        Wi = 64
        Hi = 64
        """
        x, img = data.transforms.presets.ssd.transform_test(image, short=512)
        class_IDs, scores, bounding_boxes = self.net(x)
        detections = self.get_cropped_detections(img,bounding_boxes,scores)
        return detections


    def get_detections_and_cropped_sections(self,frame,frame_d3,frame_p3):
        """
        :param frame: mxnet.NDArray
        :param frame_d3: mxnet.NDArray
        :param frame_p3: mxnet.NDArray
        :return: A pair formed of :
                        - np.array containg detected object appearence of the t frame.
                        - np.array containg cropped image of the t-3 frame of the corresponding detected object
                        - np.array containg cropped image of the t+3 frame of the corresponding detected object
        """
        x, img = data.transforms.presets.ssd.transform_test(frame, short=512)
        z, img_d3 = data.transforms.presets.ssd.transform_test(frame_d3, short=512)
        v, img_p3 = data.transforms.presets.ssd.transform_test(frame_p3, short=512)
        class_IDs, scores, bounding_boxes = self.net(x)
        detections = self.get_cropped_detections(img,bounding_boxes,scores)
        cropped_d3 = self.get_cropped_detections(img_d3,bounding_boxes,scores)
        cropped_p3 = self.get_cropped_detections(img_p3,bounding_boxes,scores)

        return detections,cropped_d3,cropped_p3




