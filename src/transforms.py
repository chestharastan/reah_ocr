import numpy as np
from PIL import Image


class BinaryTransform:
    """Convert a grayscale PIL image to black-and-white using a fixed threshold."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = img.convert("L")
        return img.point(lambda p: 255 if p > self.threshold else 0, "L")


class SkeletonTransform:
    """Thin strokes to 1-pixel width using morphological skeletonization (scikit-image)."""

    def __call__(self, img):
        arr = np.array(img.convert("L")) > 128
        from skimage.morphology import skeletonize
        skel = skeletonize(arr).astype(np.uint8) * 255
        return Image.fromarray(skel, mode="L")
