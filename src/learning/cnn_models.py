# Author: Mikita Sazanovich

import tensorflow as tf
from src.utils.config import CFG


class CNNNetwork:
    def __init__(self, reg_coef):
        self.reg_coef = reg_coef

        self.x = None
        self.is_train = None
        self.true_output = None
        self.__setup_inputs()
        assert self.x is not None
        assert self.is_train is not None
        assert self.true_output is not None

        self.output = None
        self.__setup_output()
        assert self.output is not None

        self.loss = None
        self.__setup_loss()
        assert self.loss is not None

        self.train_op = None

    def __setup_inputs(self):
        # define placeholder variable for input images
        self.x = tf.placeholder(tf.float32, shape=[None, CFG.image_height * CFG.image_width], name='x')
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # define placeholder for the true omega velocities
        # [None: tensor may hold arbitrary num of velocities, number of omega predictions for each image]
        self.true_output = tf.placeholder(tf.float32, shape=[None, 2], name="true_output")

    def __setup_output(self):
        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            # define the 4-d tensor expected by TensorFlow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(self.x, [-1, CFG.image_height, CFG.image_width, 1])

            # define 1st convolutional layer
            hl_conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=2, padding="valid",
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_coef),
                                         name="conv_layer_1")
            hl_conv_1 = tf.layers.batch_normalization(hl_conv_1, training=self.is_train)
            hl_conv_1 = tf.nn.relu(hl_conv_1)

            max_pool_1 = tf.layers.max_pooling2d(hl_conv_1, pool_size=2, strides=2)

            # define 2nd convolutional layer
            hl_conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=8, padding="valid",
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_coef),
                                         name="conv_layer_2")
            hl_conv_2 = tf.layers.batch_normalization(hl_conv_2, training=self.is_train)
            hl_conv_2 = tf.nn.relu(hl_conv_2)

            max_pool_2 = tf.layers.max_pooling2d(hl_conv_2, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            conv_flat = tf.layers.flatten(max_pool_2)

            # add 1st fully connected layers to the neural network
            hl_fc_1 = tf.layers.dense(inputs=conv_flat, units=64, name="fc_layer_1",
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_coef))
            hl_fc_1 = tf.layers.batch_normalization(hl_fc_1, training=self.is_train)
            hl_fc_1 = tf.nn.relu(hl_fc_1)

            # add 2nd fully connected layers to predict the driving commands
            hl_fc_2 = tf.layers.dense(inputs=hl_fc_1, units=2, name="fc_layer_2",
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_coef))
            # no batch norm for the final layer
            hl_fc_2 = tf.nn.tanh(hl_fc_2)
            self.output = hl_fc_2

    def __setup_loss(self):
        with tf.name_scope("Loss"):
            task_loss = tf.reduce_mean(tf.square(self.true_output - self.output))
            reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
            loss = task_loss + reg_loss
            self.loss = loss

    def add_train_op(self, learning_rate):
        with tf.name_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            self.train_op = train_op
