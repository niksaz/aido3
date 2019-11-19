# Author: Mikita Sazanovich

import cv2
import numpy as np
from src.utils.config import CFG


def preprocess_image(image):
    # crop the 1/3 upper part of the image
    new_img = image[int(image.shape[0] / 3):, :, :]

    # transform the color image to grayscale
    new_img = cv2.cvtColor(new_img[:, :, :], cv2.COLOR_RGB2GRAY)

    # resize the image to the expected size
    new_img = cv2.resize(new_img, (CFG.image_width, CFG.image_height))

    return new_img


def prepare_for_the_model(image):
    image = image.astype(np.float32)
    image = image / 255
    return image
