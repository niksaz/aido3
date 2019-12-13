# Author: Mikita Sazanovich

from abc import ABC, abstractmethod

import tensorflow as tf

from src.utils.config import CFG


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


class CNNModelBase(ABC):

    @abstractmethod
    def setup_output(self):
        pass

    @staticmethod
    @abstractmethod
    def get_output_node_from_graph(graph):
        pass

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

    def create_feed_dict(self, X, early_drop_prob, late_drop_prob, is_train, Y=None):
        feed_dict = {
            self.x: X,
            self.batch_size: len(X),
            self.early_drop_prob: early_drop_prob,
            self.late_drop_prob: late_drop_prob,
            self.is_train: is_train,
        }
        if Y is not None:
            feed_dict[self.true_output] = Y
        return feed_dict

    @staticmethod
    def create_feed_dict_from_graph(graph, X):
        tensor_x = graph.get_tensor_by_name('prefix/x:0')
        tensor_batch_size = graph.get_tensor_by_name('prefix/batch_size:0')
        tensor_early_drop_prob = graph.get_tensor_by_name('prefix/early_drop_prob:0')
        tensor_late_drop_prob = graph.get_tensor_by_name('prefix/late_drop_prob:0')
        tensor_is_train = graph.get_tensor_by_name('prefix/is_train:0')
        feed_dict = {
            tensor_x: X,
            tensor_batch_size: len(X),
            tensor_early_drop_prob: 0.0,
            tensor_late_drop_prob: 0.0,
            tensor_is_train: False,
        }
        return feed_dict

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
                X, units=16, name='fc_layer_1',
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                kernel_constraint=tf.keras.constraints.max_norm(max_value=CFG.max_norm_value, axis=0),
                activation=tf.nn.relu)

            with tf.variable_scope('IC_fc_layer_2', reuse=tf.AUTO_REUSE):
                X = tf.layers.batch_normalization(X, axis=1, training=self.is_train, name='batch_norm')
                X = tf.nn.dropout(X, rate=self.late_drop_prob, name='dropout')

            X = tf.layers.dense(
                X, units=2, name='fc_layer_2',
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed),
                kernel_constraint=tf.keras.constraints.max_norm(max_value=CFG.max_norm_value, axis=0))
            output = X

        self.output = output

    @staticmethod
    def get_output_node_from_graph(graph):
        Y = graph.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')
        return Y

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
                kernel_constraint=tf.keras.constraints.max_norm(max_value=CFG.max_norm_value, axis=[0, 1, 2]),
                activation=tf.nn.relu)
            output = conv2d
        return output
