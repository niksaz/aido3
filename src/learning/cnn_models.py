# Author: Mikita Sazanovich

from abc import ABC, abstractmethod

import tensorflow as tf

from src.utils.config import CFG


class CNNModelBase(ABC):

    def setup_inputs(self):
        # define placeholder variable for input images
        self.x = tf.placeholder(tf.float32,
                                shape=[None, CFG.image_height, CFG.image_width, 3],
                                name='x')
        self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
        self.early_drop_prob = tf.placeholder(tf.float32, shape=(), name='early_drop_prob')
        self.late_drop_prob = tf.placeholder(tf.float32, shape=(), name='late_drop_prob')
        self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

        # define placeholder for the true omega velocities
        # [None: tensor may hold arbitrary num of velocities, number of omega predictions for each image]
        self.true_output = tf.placeholder(tf.float32, shape=[None, 2], name="true_output")

    @abstractmethod
    def setup_output(self):
        pass

    def setup_loss(self):
        with tf.name_scope("Loss"):
            self.task_loss = tf.reduce_mean(tf.square(self.true_output - self.output))
            self.reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
            self.loss = self.task_loss + self.reg_loss

    def add_train_op(self):
        with tf.name_scope("optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.get_scheduled_learning_rate(),
                    beta1=CFG.optimizer_adam_beta1,
                    beta2=CFG.optimizer_adam_beta2,
                    epsilon=CFG.optimizer_adam_epsilon)
                self.train_op = self.optimizer.minimize(
                    self.loss,
                    global_step=tf.train.get_or_create_global_step())

    @staticmethod
    def get_scheduled_learning_rate():
        lr = CFG.learning_rate
        warmup_steps = CFG.learning_rate_warmup_steps
        with tf.name_scope("learning_rate"):
            warmup_steps = tf.to_float(warmup_steps)
            step = tf.to_float(tf.train.get_or_create_global_step())

            # Apply linear warmup
            lr *= tf.minimum(1.0, step / warmup_steps)
            # Apply rsqrt decay
            lr *= tf.rsqrt(tf.maximum(step, warmup_steps))

            # Create a named tensor that will be logged using the logging hook.
            # The full name includes variable and names scope. In this case, the name
            # is model/get_train_op/learning_rate/learning_rate
            tf.identity(lr, "learning_rate")

            return lr


class CNNResidualNetwork(CNNModelBase):

    def __init__(self):
        self.setup_inputs()
        self.setup_output()
        self.setup_loss()

    def _residual_block(self, x, size, dropout=False, dropout_prob=0.5, seed=None):
        residual = tf.layers.batch_normalization(x, training=self.is_train)
        residual = tf.nn.relu(residual)
        residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, strides=2, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                    kernel_regularizer=tf.keras.regularizers.l2(CFG.regularizer))
        if dropout:
            residual = tf.nn.dropout(residual, dropout_prob, seed=seed)
        residual = tf.layers.batch_normalization(residual, training=self.is_train)
        residual = tf.nn.relu(residual)
        residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, padding='same',
                                    kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                                    kernel_regularizer=tf.keras.regularizers.l2(CFG.regularizer))
        if dropout:
            residual = tf.nn.dropout(residual, dropout_prob, seed=seed)

        return residual

    def _one_residual(self, x, keep_prob=0.5, seed=None):
        nn = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                              kernel_regularizer=tf.keras.regularizers.l2(CFG.regularizer))
        nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

        rb_1 = self._residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)

        nn = tf.layers.conv2d(nn, filters=32, kernel_size=1, strides=2, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                              kernel_regularizer=tf.keras.regularizers.l2(CFG.regularizer))
        nn = tf.keras.layers.add([rb_1, nn])

        nn = tf.layers.flatten(nn)

        return nn

    def setup_output(self):
        seed = CFG.seed

        x_shaped = tf.reshape(self.x, [-1, CFG.image_height, CFG.image_width, 1])
        model = self._one_residual(x_shaped, seed=seed)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))

        model = tf.layers.dense(model, 2)

        self.output = model


class CNN160Model(CNNModelBase):

    def __init__(self):
        self.setup_inputs()
        self.setup_output()
        self.setup_loss()

    def setup_output(self):
        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            x_shaped = tf.reshape(self.x, [-1, CFG.image_height, CFG.image_width, 1])
            x_normed = tf.layers.batch_normalization(x_shaped, axis=1, training=self.is_train)

            conv_layer_block_1 = self.__build_conv_block(x_normed, kernel_size=3, filters=4, name='conv1')
            conv_layer_block_2 = self.__build_conv_block(conv_layer_block_1, kernel_size=3, filters=4, name='conv2')
            conv_layer_block_3 = self.__build_conv_block(conv_layer_block_2, kernel_size=3, filters=8, name='conv3')
            conv_layer_block_4 = self.__build_conv_block(conv_layer_block_3, kernel_size=3, filters=16, name='conv4')

            conv_flat = tf.layers.flatten(conv_layer_block_4)

            # add 1st fully connected layers to the neural network
            hl_fc_1 = tf.layers.dense(inputs=conv_flat, units=64, name="fc_layer_1",
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=CFG.regularizer))
            hl_fc_1 = tf.layers.batch_normalization(hl_fc_1, axis=-1, training=self.is_train)
            hl_fc_1 = tf.nn.tanh(hl_fc_1)

            # add 2nd fully connected layers to predict the driving commands
            hl_fc_2 = tf.layers.dense(inputs=hl_fc_1, units=2, name="fc_layer_2",
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=CFG.regularizer))
            hl_fc_2 = tf.nn.tanh(hl_fc_2)

            self.output = hl_fc_2

    def __build_conv_block(self, input_layer, kernel_size, filters, name):
        with tf.variable_scope(name):
            conv2d_layer = tf.layers.conv2d(input_layer, kernel_size=kernel_size, filters=filters, padding="same",
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=CFG.regularizer))
            batch_norm_layer = tf.layers.batch_normalization(conv2d_layer, axis=1, training=self.is_train)
            non_linear_layer = tf.nn.relu(batch_norm_layer)
            max_pool_layer = tf.layers.max_pooling2d(non_linear_layer, pool_size=2, strides=2, padding="same")
            return max_pool_layer


class CNN96Model(CNNModelBase):

    def __init__(self):
        self.setup_inputs()
        self.setup_output()
        self.setup_loss()

    def setup_output(self):
        seed = CFG.seed
        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
            X = self.x

            X = self.__add_conv_branch(
                X, kernel_size=5, filters=2, drop_prob=self.early_drop_prob, name='conv_1')
            X = tf.layers.max_pooling2d(
                X, pool_size=2, strides=2, name='max_pool_1')

            X = self.__add_conv_branch(
                X, kernel_size=5, filters=12, drop_prob=self.late_drop_prob, name='conv_2')
            X = tf.layers.max_pooling2d(
                X, pool_size=2, strides=2, name='max_pool_2')

            X = self.__add_conv_branch(
                X, kernel_size=5, filters=24, drop_prob=self.late_drop_prob, name='conv_3')
            X = tf.layers.max_pooling2d(
                X, pool_size=2, strides=2, name='max_pool_3')

            X = self.__add_conv_branch(
                X, kernel_size=5, filters=36, drop_prob=self.late_drop_prob, name='conv_4')
            X = tf.layers.max_pooling2d(
                X, pool_size=2, strides=2, name='max_pool_4')

            X = self.__add_conv_branch(
                X, kernel_size=5, filters=48, drop_prob=self.late_drop_prob, name='conv_5')
            X = tf.layers.max_pooling2d(
                X, pool_size=2, strides=2, name='max_pool_5')

            X = tf.layers.flatten(X, name='conv_flat')

            with tf.variable_scope('IC_fc_layer_1', reuse=tf.AUTO_REUSE):
                X = tf.layers.batch_normalization(X, axis=1, training=self.is_train, name='batch_norm')
                X = tf.nn.dropout(X, rate=self.late_drop_prob, name='dropout')

            X = tf.layers.dense(
                X, units=64, name='fc_layer_1',
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                kernel_constraint=tf.keras.constraints.max_norm(max_value=4, axis=0),
                activation=tf.nn.relu)

            with tf.variable_scope('IC_fc_layer_2', reuse=tf.AUTO_REUSE):
                X = tf.layers.batch_normalization(X, axis=1, training=self.is_train, name='batch_norm')
                X = tf.nn.dropout(X, rate=self.late_drop_prob, name='dropout')

            X = tf.layers.dense(
                X, units=2, name='fc_layer_2',
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                kernel_constraint=tf.keras.constraints.max_norm(max_value=4, axis=0))
            output = X

        self.output = output

    def __build_spatial_dropout(self, x, drop_prob, name):
        """
        A function for 2D spatial dropout.
        :param x: is a tensor of shape [batch_size, height, width, channels]
        """
        input_shape = x.get_shape().as_list()
        channels = input_shape[3]

        x_drop = tf.nn.dropout(x, rate=drop_prob, noise_shape=[self.batch_size, 1, 1, channels], name=name)
        output = x_drop
        return output

    def __add_conv_branch(self, x, kernel_size, filters, drop_prob, name):
        seed = CFG.seed
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # Independent-Component (IC) layer
            batch_norm = tf.layers.batch_normalization(
                x, axis=3, training=self.is_train, name='batch_norm')
            dropped = self.__build_spatial_dropout(batch_norm, drop_prob=drop_prob, name='dropout')
            # Weight+Activation layer
            conv2d = tf.layers.conv2d(
                dropped, kernel_size=kernel_size, filters=filters, padding='same', name='conv2d',
                kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
                kernel_constraint=tf.keras.constraints.max_norm(max_value=4, axis=[0, 1, 2]),
                activation=tf.nn.relu)
            output = conv2d
        return output
