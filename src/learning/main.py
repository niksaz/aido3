#!/usr/bin/env python

import argparse
import logging
import os
import shutil
import time

from src.learning.cnn_models import CNN96Model
from src.learning.cnn_training_functions import Trainer
from src.learning.dataset import ConcreteTrainingDataset, CombinedTrainingDataset
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

    datasets = []
    for dataset_name in ['jetbrains', 'duckietown', 'loop_empty', 'udem1']:
        dataset = ConcreteTrainingDataset(os.path.join('data', dataset_name), CFG.train_data_ratio, CFG.seed)
        datasets.append(dataset)
        logger.info(f"{dataset_name} dataset's len is {len(dataset)}")
    dataset = CombinedTrainingDataset(datasets)

    # create and train the model
    model = CNN96Model()

    start_time = time.time()
    trainer = Trainer(CFG.batch_size)
    trainer.train(model, model_dir, dataset, CFG.steps_to_train_for)

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60
    logger.info('Finished training of {} in {:.1f} minutes'.format(args.model_name, training_time))


if __name__ == '__main__':
    main()
