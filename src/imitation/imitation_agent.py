#!/usr/bin/env python3

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from aido_schemas import EpisodeStart, protocol_agent_duckiebot1, PWMCommands, Duckiebot1Commands, LEDSCommands, RGB, \
    wrap_direct, Context, Duckiebot1Observations, JPGImage

from src.learning.cnn_models import load_graph, CNN96Model
from src.utils.preprocessing import preprocess_image, prepare_for_the_model


@dataclass
class ImitationAgentConfig:
    # pwm_left_interval: Tuple[int, int] = (0.2, 0.3)
    # pwm_right_interval: Tuple[int, int] = (0.2, 0.3)
    current_image: np.ndarray = np.zeros((480, 640))


class ImitationAgent:
    config: ImitationAgentConfig = ImitationAgentConfig()

    def init(self, context: Context):
        print(time.time(), 'init')
        context.info('init()')

        frozen_model_filename = "frozen_graph.pb"

        # We use our "load_graph" function to load the graph
        self.graph = load_graph(frozen_model_filename)
        self.session = tf.Session(graph=self.graph)

    def on_received_seed(self, data: int):
        print(time.time(), 'on_received_seed')
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        print(time.time(), 'on_received_episode_start')
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: Duckiebot1Observations):
        print(time.time(), 'on_received_observations')
        camera: JPGImage = data.camera
        self.config.current_image = jpg2rgb(camera.jpg_data)

    def compute_action(self, observation):
        print(time.time(), 'compute_action')
        observation = prepare_for_the_model(preprocess_image(observation))

        X = observation
        X = np.expand_dims(X, axis=0)

        action = self.session.run(
            CNN96Model.get_output_node_from_graph(self.graph),
            CNN96Model.create_feed_dict_from_graph(self.graph, X))
        action = [action[0, 0], action[0, 1]]

        return action

    def on_received_get_commands(self, context: Context):
        print(time.time(), 'on_received_get_commands')
        # l, u = self.config.pwm_left_interval
        # pwm_left = np.random.uniform(l, u)
        # l, u = self.config.pwm_right_interval
        # pwm_right = np.random.uniform(l, u)
        pwm_left, pwm_right = self.compute_action(self.config.current_image)
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))
        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=float(pwm_left),
                                   motor_right=float(pwm_right))
        commands = Duckiebot1Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        print(time.time(), 'finish')
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = ImitationAgent()
    protocol = protocol_agent_duckiebot1
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
