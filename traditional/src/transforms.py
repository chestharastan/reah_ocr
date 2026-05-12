import numpy as np
from PIL import Image
import cv2

class BinaryTransform:
    """Convert a grayscale PIL image to black-and-white using a fixed threshold."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = img.convert("L")
        return img.point(lambda p: 255 if p > self.threshold else 0, "L")


class SkeletonTransform:
    """Thin strokes to 1-pixel width using morphological skeletonization (OpenCV)."""
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
        
        skel = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()
        
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            subset = cv2.subtract(temp, opened)
            skel = cv2.bitwise_or(skel, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        
        return Image.fromarray(skel, mode="L")