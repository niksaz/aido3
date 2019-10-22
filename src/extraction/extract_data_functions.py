#!/usr/bin/env python

import numpy as np
import pandas as pd
from copy import copy
import cv2

def image_preprocessing(image):

    # define new image's height and width
    image_final_height = 48
    image_final_width = 96

    # crop the upper third of the image -> new image will be 320x640
    new_img = image[int(image.shape[0]/3):, :, :]

    # transform the color image to grayscale
    new_img = cv2.cvtColor(new_img[:, :, :], cv2.COLOR_RGB2GRAY)

    # resize the image from 320x640 to 48x96
    new_img = cv2.resize( new_img, (image_final_width, image_final_height) ) # this returns image 48x96 and not 96x48

    # normalize images to range [0, 1] (divide each pixel by 255)
    # first transform the array of int to array of float, else the division with 255 will return an array of 0s
    new_img = new_img.astype(float)
    new_img = new_img / 255

    # new_part
    new_img = np.reshape(new_img, (1, -1))

    return new_img


def synchronize_data(df_imgs, df_cmds, bag_ID):
    # initialize a dataframe to append all new values
    synch_data = pd.DataFrame()

    first_time = True

    # for each omega velocity, find the respective image
    for cmd_index, cmd_time in enumerate(df_cmds['vel_timestamp']):

        # we keep only the data for which the duckiebot is moving (we do not want the duckiebot to learn to remain at rest)
        if ( df_cmds['vel_left'][cmd_index] != 0) & ( df_cmds['vel_right'][cmd_index] != 0):

            # find index of image with the closest timestamp to wheels' velocities timestamp
            img_index = ( np.abs( df_imgs['img_timestamp'].values - cmd_time ) ).argmin()

            # The image precedes the omega velocity, thus image's timestamp must be smaller
            if ( ( df_imgs['img_timestamp'][img_index] - cmd_time ) > 0 ) & (img_index - 1 < 0):

                # if the image appears after the velocity and there is no previous image, then
                # there is no safe synchronization and the data should not be included
                continue
            else:

                # if the image appears after the velocity, in this case we know that there is previous image and we
                # should prefer it
                if ( df_imgs['img_timestamp'][img_index] - cmd_time ) > 0 :

                    img_index = img_index - 1

                # create a numpy array for all data except the images
                temp_data = np.array( [[
                    df_imgs['img_timestamp'][img_index],
                    df_cmds["vel_timestamp"][cmd_index],
                    df_cmds['vel_left'][cmd_index],
                    df_cmds['vel_right'][cmd_index],
                    bag_ID
                ]] )

                # create a new numpy array only for images (images are row vectors of size (1,4608) and it is more
                # convenient to save them separately
                temp_imgs = df_imgs['img'][img_index]

                if first_time:

                    synch_data = copy(temp_data)
                    synch_imgs = copy(temp_imgs)
                    first_time = False

                else:

                    synch_data = np.vstack((synch_data, temp_data))
                    synch_imgs = np.vstack((synch_imgs, temp_imgs))


    print("Synchronization of {}.bag file is finished. From the initial {} images and {} velocities commands, the extracted "
          "synchronized data are {}.".format(bag_ID, df_imgs.shape[0], df_cmds.shape[0], synch_data.shape[0]))

    # return the synchronized data to the main function
    return synch_data, synch_imgs
