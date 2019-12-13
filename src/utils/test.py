#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.extraction.sim.env import launch_env
from src.learning.cnn_models import load_graph, CNN96Model
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

    with tf.Session(graph=graph) as session:
        STEPS = 256
        env = launch_env(map_name=args.map_name)

        while True:
            observation = env.reset()
            for steps in range(STEPS):
                env.render()

                X = prepare_for_the_model(preprocess_image(observation))
                X = np.expand_dims(X, axis=0)

                action = session.run(
                    CNN96Model.get_output_node_from_graph(graph),
                    CNN96Model.create_feed_dict_from_graph(graph, X))
                action = np.array([action[0, 0], action[0, 1]])
                action = np.clip(action, -1.0, 1.0)
                print(f'Taking action: {action}')

                observation, reward, done, info = env.step(action)
                if done:
                    break

        env.close()


if __name__ == '__main__':
    main()
