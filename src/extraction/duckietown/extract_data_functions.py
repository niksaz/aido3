#!/usr/bin/env python

import numpy as np


def synchronize_data(df_imgs, df_cmds, bag_ID):
    # initialize a dataframe to append all new values
    synch_data = []
    synch_imgs = []

    # for each omega velocity, find the respective image
    for cmd_index, cmd_time in enumerate(df_cmds['vel_timestamp']):
        # we keep only the data for which the duckiebot is moving (we do not want the duckiebot to learn to remain at rest)
        if (df_cmds['vel_left'][cmd_index] != 0) & (df_cmds['vel_right'][cmd_index] != 0):

            # find index of image with the closest timestamp to wheels' velocities timestamp
            img_index = (np.abs(df_imgs['img_timestamp'].values - cmd_time)).argmin()

            # The image precedes the omega velocity, thus image's timestamp must be smaller
            if ((df_imgs['img_timestamp'][img_index] - cmd_time) > 0) & (img_index - 1 < 0):

                # if the image appears after the velocity and there is no previous image, then
                # there is no safe synchronization and the data should not be included
                continue
            else:

                # if the image appears after the velocity, in this case we know that there is previous image and we
                # should prefer it
                if (df_imgs['img_timestamp'][img_index] - cmd_time) > 0:
                    img_index = img_index - 1

                # create a numpy array for all data except the images
                temp_data = [
                    df_imgs['img_timestamp'][img_index],
                    df_cmds["vel_timestamp"][cmd_index],
                    df_cmds['vel_left'][cmd_index],
                    df_cmds['vel_right'][cmd_index],
                    bag_ID,
                ]

                # create a new numpy array only for images (images are row vectors of size (1,4608) and it is more
                # convenient to save them separately
                temp_img = df_imgs['img'][img_index]

                synch_data.append(temp_data)
                synch_imgs.append(temp_img)

    print(
        "Synchronization of {}.bag file is finished. From the initial {} images and {} velocities commands, the extracted "
        "synchronized data are {}.".format(bag_ID, df_imgs.shape[0], df_cmds.shape[0], len(synch_data)))

    # return the synchronized data to the main function
    return synch_data, synch_imgs
