import cv2

from code.training.utils.ObjectDetector import ObjectDetector

video_path = '/home/george/Downloads/Licenta-refactored/Avenue Dataset/testing_videos/11.avi'
video = cv2.VideoCapture(video_path)
ret, frame = video.read()
object_detector = ObjectDetector(frame)
images =object_detector.get_object_detections()
for image in images:
    cv2.imshow('windows',image)
    cv2.waitKey()
    cv2.destroyAllWindows()
