#!/usr/env/python

import cv2
import numpy as np


def fun_img_preprocessing(image, image_final_height, image_final_width):
    # crop the 1/3 upper part of the image
    new_img = image[int(image.shape[0] / 3):, :, :]

    # transform the color image to grayscale
    new_img = cv2.cvtColor(new_img[:, :, :], cv2.COLOR_RGB2GRAY)

    # resize the image from 320x640 to 48x96
    new_img = cv2.resize(new_img, (image_final_width, image_final_height))

    # normalize images to range [0, 1] (divide each pixel by 255)
    # first transform the array of int to array of float else the division with 255 will return an array of 0s
    new_img = new_img.astype(float)
    new_img = new_img / 255

    # new_part
    new_img = np.reshape(new_img, (1, -1))

    return new_img
