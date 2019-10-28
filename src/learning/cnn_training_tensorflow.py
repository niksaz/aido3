#!/usr/bin/env python

import time
import os
from cnn_training_functions import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():

    # define path for training dataset
    file_data_path = os.path.join(os.getcwd(), 'data', 'LF_dataset.h5')

    # define batch_size (e.g 50, 100)
    batch_size = 64

    # define which optimizer you want to use (e.g "Adam", "GDS"). For "Adam" and "GDS" this script will take care the rest.
    # ATTENTION !! If you want to choose a different optimizer from these two, you will have to add it in the training functions.
    optimizer = "GDS"

    # define learning rate (e.g 1E-3, 1E-4, 1E-5):
    learning_rate = 1E-4

    # define total epochs (e.g 1000, 5000, 10000)
    epochs = 1000

    # read train data
    print('Reading train dataset')
    train_velocities, train_images = load_data(file_data_path, "training")

    # read test data
    print('Reading test dataset')
    test_velocities, test_images = load_data(file_data_path, "testing")

    # construct model name based on the hyper parameters
    # model_name = form_model_name(batch_size, learning_rate, optimizer, epochs)
    model_name = 'learned_models'

    print('Starting training for {} model.'.format(model_name))

    # keep track of training time
    start_time = time.time()

    # train model
    cnn_train = CNN_training(batch_size, epochs, learning_rate, optimizer)
    cnn_train.training(model_name, train_velocities, train_images, test_velocities, test_images)

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60

    print('Finished training of {} in {} minutes.'.format(model_name, training_time))


if __name__ == '__main__':
    main()
