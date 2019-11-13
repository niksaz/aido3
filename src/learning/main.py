#!/usr/bin/env python

import argparse
import os
import time

import sklearn.model_selection

from src.learning.cnn_models import CNNX2Model, CNNX4Model
from src.learning.cnn_training_functions import load_sim_data, Trainer
from src.utils.config import CFG

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='The name of the trained model')
    parser.add_argument('--seed', type=int, help='Random seed to split the data with')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    # # define path for training dataset
    # real_file_data_path = os.path.join(os.getcwd(), 'data', 'LF_dataset.h5')
    #
    # # read train data
    # print('Reading train dataset')
    # train_velocities, train_images = load_real_data(real_file_data_path, "training")
    #
    # # read test data
    # print('Reading test dataset')
    # test_velocities, test_images = load_real_data(real_file_data_path, "testing")

    sim_file_data_path = os.path.join(os.getcwd(), 'data', 'LF_dataset_sim.h5')
    velocities, images = load_sim_data(sim_file_data_path)
    train_velocities, test_velocities, train_images, test_images = sklearn.model_selection.train_test_split(
        velocities, images, train_size=0.7, random_state=args.seed)

    # construct the model name
    model_dir = 'learned_models'
    model_name = args.model_name

    print('Starting training for {} model.'.format(model_name))

    # keep track of training time
    start_time = time.time()

    # create and train the model
    if CFG.model == 'CNNX2Model':
        model = CNNX2Model(CFG.regularizer)
    elif CFG.model == 'CNNX4Model':
        model = CNNX4Model(CFG.regularizer)
    else:
        raise ValueError(f'Unknown model from the config: {format(CFG.model)}')

    trainer = Trainer(CFG.batch_size, CFG.epochs, CFG.lr)
    trainer.train(model, model_dir, model_name, train_velocities, train_images, test_velocities, test_images)

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60

    print('Finished training of {} in {} minutes.'.format(model_name, training_time))


if __name__ == '__main__':
    main()
