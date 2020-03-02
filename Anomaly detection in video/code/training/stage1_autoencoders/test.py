import cv2
import mxnet as mx
import numpy as np
from gluoncv import utils
from matplotlib import pyplot as plt

from code.training.utils.ObjectDetector import ObjectDetector

video_path = "/Avenue Dataset/training_videos_small/01.avi"
video = cv2.VideoCapture(video_path)

ret,frame = video.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = mx.nd.array(frame)
frame = frame.astype(np.uint8)

objectDetector = ObjectDetector(frame)

ax = utils.viz.plot_bbox(objectDetector.img_transformed_image, objectDetector.bounding_boxes, objectDetector.scores)

plt.show()
