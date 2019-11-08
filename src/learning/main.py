#!/usr/bin/env python

import time
import os
import argparse
from src.learning.cnn_training_functions import load_data, Trainer
from src.learning.cnn_models import CNNX4Model
from src.utils.config import CFG

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='The name of the trained model')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    # define path for training dataset
    file_data_path = os.path.join(os.getcwd(), 'data', 'LF_dataset.h5')

    # read train data
    print('Reading train dataset')
    train_velocities, train_images = load_data(file_data_path, "training")

    # read test data
    print('Reading test dataset')
    test_velocities, test_images = load_data(file_data_path, "testing")

    # construct the model name
    model_dir = 'learned_models'
    model_name = args.model_name

    print('Starting training for {} model.'.format(model_name))

    # keep track of training time
    start_time = time.time()

    # create and train the model
    model = CNNX4Model(CFG.regularizer)
    trainer = Trainer(CFG.batch_size, CFG.epochs, CFG.lr)
    trainer.train(model, model_dir, model_name, train_velocities, train_images, test_velocities, test_images)

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60

    print('Finished training of {} in {} minutes.'.format(model_name, training_time))


if __name__ == '__main__':
    main()
