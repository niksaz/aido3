#!/usr/env/python

import cv2
import numpy as np
# from mvnc import mvncapi as mvnc


def fun_img_preprocessing(image, image_final_height, image_final_width):

    # crop the 1/3 upper part of the image
    new_img = image[int(image.shape[0]/3):, :, :]

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


# def load_movidius_graph(path_to_graph):
#
#     # find movidius stick devices
#     devices = mvnc.enumerate_devices()
#     if len(devices) == 0:
#         print('No devices found')
#         quit()
#
#     # get movidius stick device
#     device = mvnc.Device(devices[0])
#
#     # open movidius stick
#     device.open()
#
#     # Load graph
#     with open(path_to_graph, mode='rb') as f:
#         graphFileBuff = f.read()
#
#     graph = mvnc.Graph(path_to_graph)
#     fifoIn, fifoOut = graph.allocate_with_fifos(device, graphFileBuff)
#
#     return graph, fifoIn, fifoOut
#
#
# def movidius_cnn_predictions(graph, fifoIn, fifoOut, img):
#
#     # run CNN
#     graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img.astype(np.float32), 'user object')
#     output, userobj = fifoOut.read_elem()
#
#     return output[0]
#
#
# def destroy_all(object):
#
#     # close fifo queues, graph and device
#     object.fifoIn.destroy()
#     object.fifoOut.destroy()
#     object.graph.destroy()
#     object.device.close()
