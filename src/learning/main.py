#!/usr/bin/env python

import argparse
import logging
import os
import shutil
import time

from src.learning.cnn_models import CNNResidualNetwork, CNN160Model, CNN96Model
from src.learning.cnn_training_functions import Trainer
from src.learning.dataset import CombinedDataset, Dataset
from src.utils.config import CFG

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger()


def configure_logging(model_dir):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    filename = os.path.join(model_dir, 'log.txt')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='The name of the trained model')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    model_dir = os.path.join('learned_models', args.model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    configure_logging(model_dir)

    real_dataset = Dataset(os.path.join('data', 'real'))
    sim_dataset = Dataset(os.path.join('data', 'sim'))
    dataset = CombinedDataset([real_dataset, sim_dataset])
    logger.info(f'Real dataset size is {len(real_dataset)}')
    logger.info(f'Sim dataset size is {len(sim_dataset)}')
    logger.info(f'CombinedDataset size is {len(dataset)}')

    logger.info('Starting training for {} model'.format(args.model_name))
    start_time = time.time()

    # create and train the model
    if CFG.model == 'CNNResidualNetwork':
        model = CNNResidualNetwork(CFG.regularizer)
    elif CFG.model == 'CNN160Model':
        model = CNN160Model(CFG.regularizer)
    elif CFG.model == 'CNN96Model':
        model = CNN96Model(CFG.regularizer)
    else:
        raise ValueError(f'Unknown model from the config: {format(CFG.model)}')

    trainer = Trainer(CFG.batch_size, CFG.epochs, CFG.lr)
    trainer.train(model, model_dir, dataset)

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60

    logger.info('Finished training of {} in {:.1f} minutes'.format(args.model_name, training_time))


if __name__ == '__main__':
    main()
