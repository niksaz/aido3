#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import gym
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.imitation.graph_utils import load_graph
from src.utils.preprocessing import preprocess_image, prepare_for_the_model


def launch_env(id=None):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123,  # random seed
            map_name="loop_empty",
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env


def main():
    frozen_model_filename = "frozen_graph.pb"

    graph = load_graph(frozen_model_filename)
    x = graph.get_tensor_by_name('prefix/x:0')
    batch_size = graph.get_tensor_by_name('prefix/batch_size:0')
    early_drop_prob = graph.get_tensor_by_name('prefix/early_drop_prob:0')
    late_drop_prob = graph.get_tensor_by_name('prefix/late_drop_prob:0')
    is_train = graph.get_tensor_by_name('prefix/is_train:0')
    y = graph.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')

    with tf.Session(graph=graph) as sess:
        env = launch_env()
        STEPS = 1024

        observation = env.reset()
        for steps in range(STEPS):
            env.render()

            X = prepare_for_the_model(preprocess_image(observation, cv2.COLOR_BGR2RGB))
            X = np.expand_dims(X, axis=0)

            action = sess.run(
                y,
                feed_dict={
                    x: X,
                    batch_size: 1,
                    early_drop_prob: 0.0,
                    late_drop_prob: 0.0,
                    is_train: False,
                })
            action = np.array([action[0, 0], action[0, 1]])
            print(f'Taking action: {action}')

            observation, reward, done, info = env.step(action)
            if done:
                break
        env.close()


if __name__ == '__main__':
    main()
