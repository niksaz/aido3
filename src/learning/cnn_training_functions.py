# Author: Mikita Sazanovich

import logging
import os
import time

import numpy as np
import tensorflow as tf

from src.learning.cnn_models import CNNModelBase
from src.learning.dataset import CombinedTrainingDataset, TrainingDataset
from src.utils.config import CFG

logger = logging.getLogger()


class Trainer:

    @staticmethod
    def __get_learning_rate_for_epoch(epoch):
        lr = CFG.learning_rate
        for decay_epoch in CFG.lr_decay_epochs:
            if epoch >= decay_epoch:
                lr *= 1e-1
        return lr

    def __init__(self, batch_size: int, epochs: int):
        self.batch_size = batch_size
        self.epochs = epochs
        self.sess = None

    def __run_epoch_for(self, epoch: int, model: CNNModelBase, dataset: TrainingDataset, mode: str):
        """
        For each epoch extract batches and execute train or test step depending on the mode

        :param mode: 'train' or 'test' in order to define if backpropagation is executed as well or not
        :return: mean of losses for the epoch
        """
        if mode == 'train':
            data_indices = dataset.get_train_indices()
            np.random.shuffle(data_indices)
        elif mode == 'test':
            data_indices = dataset.get_test_indices()
            # No need to shuffle in a testing stage
        else:
            raise NotImplementedError('Unknown mode: {}'.format(mode))
        lr = self.__get_learning_rate_for_epoch(epoch)

        batch_losses = []
        for i in range(0, len(data_indices), self.batch_size):
            time_started = time.time()
            X = []
            Y = []
            for index in data_indices[i:i + self.batch_size]:
                x, y = dataset[index]
                X.append(x)
                Y.append(y)
            X = np.array(X)
            Y = np.array(Y)
            time_data_loaded = time.time()

            if mode == 'train':
                # train using the batch and calculate the loss
                _, c = self.sess.run([model.train_op, model.task_loss],
                                     feed_dict={model.x: X,
                                                model.batch_size: len(X),
                                                model.learning_rate: lr,
                                                model.early_drop_prob: CFG.early_drop_prob,
                                                model.late_drop_prob: CFG.late_drop_prob,
                                                model.is_train: True,
                                                model.true_output: Y})
            elif mode == 'test':
                # train using the batch and calculate the loss
                c = self.sess.run([model.task_loss],
                                  feed_dict={model.x: X,
                                             model.batch_size: len(X),
                                             model.early_drop_prob: 0.0,
                                             model.late_drop_prob: 0.0,
                                             model.is_train: False,
                                             model.true_output: Y})
            else:
                raise NotImplementedError('Unknown mode: {}'.format(mode))
            time_computation_done = time.time()

            batch_losses.append(c)

            if mode == 'train' and i == 0:
                logger.info(f'Spent {time_data_loaded - time_started} on data loading')
                logger.info(f'Spent {time_computation_done - time_data_loaded} on training')

        return np.mean(batch_losses)

    def train(self, model: CNNModelBase, model_dir: str, dataset: CombinedTrainingDataset):
        seed = 76
        tf.random.set_random_seed(seed)
        np.random.seed(seed)

        model_path = os.path.join(os.getcwd(), model_dir)
        logs_train_path = os.path.join(model_path, 'train')
        logs_test_path = os.path.join(model_path, 'test')
        graph_path = os.path.join(model_path, 'graph')

        man_loss_summary = tf.Summary()
        man_loss_summary.value.add(tag='Loss', simple_value=None)
        saver = tf.train.Saver(max_to_keep=10)

        model.add_train_op()

        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            # run initializer
            self.sess.run(init)

            # operation to write logs for Tensorboard
            tf_graph = self.sess.graph
            test_writer = tf.summary.FileWriter(logs_test_path, graph=tf.get_default_graph())
            test_writer.add_graph(tf_graph)

            train_writer = tf.summary.FileWriter(logs_train_path, graph=tf.get_default_graph())
            train_writer.add_graph(tf_graph)

            # IMPORTANT: this is a crucial part for compiling TensorFlow graph to a Movidius one later in the pipeline.
            # The important file to create is the 'graph.pb' which will be used to freeze the TensorFlow graph.
            # The 'graph.pbtxt' file is just the same graph in txt format in case you want to check the format of the
            # saved information.
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pbtxt', as_text=True)
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pb', as_text=False)

            best_test_mean_loss = None

            for epoch in range(self.epochs):
                # run train cycle
                avg_train_loss = self.__run_epoch_for(epoch, model, dataset, 'train')
                man_loss_summary.value[0].simple_value = avg_train_loss
                train_writer.add_summary(man_loss_summary, epoch)

                # run test cycles for every dataset
                test_losses = []
                for separate_dataset in dataset.get_datasets():
                    avg_test_loss = self.__run_epoch_for(epoch, model, separate_dataset, 'test')
                    test_losses.append(avg_test_loss)
                avg_test_loss = np.mean(test_losses)
                man_loss_summary.value[0].simple_value = avg_test_loss
                test_writer.add_summary(man_loss_summary, epoch)

                # periodically print out the learning progress
                print_inform_losses = (epoch + 1) % max(1, self.epochs // 100) == 0

                # check if it is the best model
                if best_test_mean_loss is None or best_test_mean_loss > avg_test_loss:
                    logger.info('Saving since it will be the best model up to now')
                    best_test_mean_loss = avg_test_loss
                    saver.save(self.sess, os.path.join(model_path, 'model_{:04d}_{:.9f}'.format(epoch + 1, avg_test_loss)))
                    saver.save(self.sess, os.path.join(model_path, 'best_model'))
                    print_inform_losses = True

                if print_inform_losses:
                    logger.info(f'Epoch: {(epoch + 1):04d}')
                    logger.info(f'avg_train_loss={avg_train_loss:.9f}, avg_test_loss={avg_test_loss:.9f}')
                    logger.info(f'test_losses={test_losses}')

            # close summary writers
            train_writer.close()
            test_writer.close()
