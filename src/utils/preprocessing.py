# Author: Mikita Sazanovich

import cv2
import numpy as np
from src.utils.config import CFG


def preprocess_image(image, cvt_color=None):
    # crop the 1/3 upper part of the image
    new_img = image[int(image.shape[0] / 3):, :, :]

    if cvt_color is not None:
        new_img = cv2.cvtColor(new_img[:, :, :], cvt_color)
    # should be an RGB image right now

    # resize the image to the expected size
    new_img = cv2.resize(new_img, (CFG.image_width, CFG.image_height))

    return new_img


def prepare_for_the_model(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = image.astype(np.float32)
    image = image / 255
    return image
