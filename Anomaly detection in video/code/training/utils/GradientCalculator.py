import cv2
import numpy as np


class GradientCalculator:

    def __init__(self) -> None:
        super().__init__()

    def calculate_gradient(self,image):
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
    def calculate_gradient_bulk(self,images):
        gradients = []
        for image in images:
            gradient = self.calculate_gradient(image)
            gradients.append(gradient)
        return np.array(gradients)