import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from architectures.densenet.densenet_bilstm_bigru_ctc import DenseNetBiLstmBiGruCtc


class SkeletonTransform:
    """Reduce strokes to 1-pixel centerlines via morphological thinning."""
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)

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

        skel = cv2.bitwise_not(skel)
        return Image.fromarray(skel, mode="L")


Model = DenseNetBiLstmBiGruCtc


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        SkeletonTransform(),
        transforms.ToTensor(),
    ])
