# Author: Mikita Sazanovich

import tensorflow as tf
import numpy as np
import os
import h5py
import logging

logger = logging.getLogger()


def load_real_data(file_path):
    """
    Loads images and velocities from hdf files and checks for potential mismatch in the number of images and velocities
    :param file_path: path to the hdf file from which it will extract the data
    :return: images, velocities as numpy arrays
    """
    # read dataframes
    with h5py.File(file_path, 'r') as f:
        data = f['split']['mix']
        vel_left = data['vel_left'][()]
        vel_right = data['vel_right'][()]

        velocities = np.concatenate((vel_left[:, np.newaxis], vel_right[:, np.newaxis]), axis=1)
        images = data['images'][()]

        logger.info(
            'The dataset is loaded: {} images and {} omega velocities.'.format(images.shape[0], velocities.shape[0]))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    return images, velocities


def load_sim_data(file_path):
    """
    :return: images, velocities as numpy arrays
    """
    with h5py.File(file_path, 'r') as f:
        data = f['split']['mix']
        images = data['observation'][()]
        velocities = data['action'][()]

        logger.info(
            'The dataset is loaded: {} images and {} omega velocities.'.format(images.shape[0], velocities.shape[0]))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    return images, velocities


class Trainer:
    def __init__(self, batch, epochs, learning_rate):
        self.batch_size = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sess = None

    def __run_epoch_for(self, model, data_size, x_data, y_data, mode):
        """
        For each epoch extract batches and execute train or test step depending on the inserted mode

        :param data_size: number of velocities and images
        :param x_data: images
        :param y_data: velocities
        :param mode: 'train' or 'test' in order to define if backpropagation is executed as well or not
        :return: mean of losses for the epoch
        """
        batch_losses = []
        i = 0
        while i <= data_size - 1:
            Xs = x_data[i: i + self.batch_size]
            Ys = y_data[i: i + self.batch_size]

            if mode == 'train':
                # train using the batch and calculate the loss
                _, c = self.sess.run([model.train_op, model.task_loss],
                                     feed_dict={model.x: Xs, model.batch_size: len(Xs), model.drop_prob: 0.05,
                                                model.true_output: Ys})

            elif mode == 'test':
                # train using the batch and calculate the loss
                c = self.sess.run([model.task_loss],
                                  feed_dict={model.x: Xs, model.batch_size: len(Xs), model.drop_prob: 0.0,
                                             model.true_output: Ys})

            else:
                raise NotImplementedError('Unknown mode: {}'.format(mode))

            batch_losses.append(c)
            i += self.batch_size

        return np.mean(batch_losses)

    def train(self, model, model_dir, train_velocities, train_images, test_velocities, test_images):
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

        model.add_train_op(self.learning_rate)

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
                p = np.random.permutation(len(train_images))
                train_images = train_images[p]
                train_velocities = train_velocities[p]

                # run train cycle
                avg_train_loss = self.__run_epoch_for(model, train_velocities.shape[0], train_images, train_velocities,
                                                      'train')

                # save the training loss using the manual summaries
                man_loss_summary.value[0].simple_value = avg_train_loss
                train_writer.add_summary(man_loss_summary, epoch)

                # run test cycle
                avg_test_loss = self.__run_epoch_for(model, test_velocities.shape[0], test_images, test_velocities,
                                                     'test')

                # save the test errors using the manual summaries
                man_loss_summary.value[0].simple_value = avg_test_loss
                test_writer.add_summary(man_loss_summary, epoch)

                # periodically print out the learning progress
                print_inform_losses = (epoch + 1) % (self.epochs // 100) == 0

                # check if it is the best model
                if best_test_mean_loss is None or best_test_mean_loss > avg_test_loss:
                    logger.info('Saving since it will be the best model up to now')
                    best_test_mean_loss = avg_test_loss
                    saver.save(self.sess, os.path.join(model_path, 'model_{:04d}_{:.9f}'.format(epoch + 1, avg_test_loss)))
                    saver.save(self.sess, os.path.join(model_path, 'best_model'))
                    print_inform_losses = True

                if print_inform_losses:
                    logger.info(
                        'Epoch: {:04d}, mean_train_loss = {:.9f}, mean_test_loss = {:.9f}'.format(
                            epoch + 1, avg_train_loss, avg_test_loss))

            # close summary writers
            train_writer.close()
            test_writer.close()
