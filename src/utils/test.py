#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.imitation.graph_utils import load_graph
from src.extraction.sim.env import launch_env
from src.utils.preprocessing import preprocess_image, prepare_for_the_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('map_name', type=str, help='The name of the map to test the agent on')  # loop_empty, udem1
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    frozen_model_path = os.path.join('learned_models', 'frozen_graph.pb')

    graph = load_graph(frozen_model_path)
    x = graph.get_tensor_by_name('prefix/x:0')
    batch_size = graph.get_tensor_by_name('prefix/batch_size:0')
    early_drop_prob = graph.get_tensor_by_name('prefix/early_drop_prob:0')
    late_drop_prob = graph.get_tensor_by_name('prefix/late_drop_prob:0')
    is_train = graph.get_tensor_by_name('prefix/is_train:0')
    y = graph.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')

    with tf.Session(graph=graph) as sess:
        STEPS = 256
        env = launch_env(map_name=args.map_name)

        while True:
            observation = env.reset()
            for steps in range(STEPS):
                env.render()

                X = prepare_for_the_model(preprocess_image(observation))
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
