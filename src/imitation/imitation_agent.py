#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
import numpy as np

from aido_schemas import EpisodeStart, protocol_agent_duckiebot1, PWMCommands, Duckiebot1Commands, LEDSCommands, RGB, \
    wrap_direct, Context, Duckiebot1Observations, JPGImage

from graph_utils import load_graph
from cnn_predictions import fun_img_preprocessing

@dataclass
class ImitationAgentConfig:
    # pwm_left_interval: Tuple[int, int] = (0.2, 0.3)
    # pwm_right_interval: Tuple[int, int] = (0.2, 0.3)
    current_image: np.ndarray = np.zeros((480, 640))


class ImitationAgent:
    config: ImitationAgentConfig = ImitationAgentConfig()

    def init(self, context: Context):
        context.info('init()')

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: Duckiebot1Observations):
        camera: JPGImage = data.camera
        self.config.current_image = jpg2rgb(camera.jpg_data)

    def compute_action(self, observation):
        frozen_model_filename = "frozen_graph.pb"

        # We use our "load_graph" function
        graph = load_graph(frozen_model_filename)

        # To check which operations your network is using
        # uncomment the following commands:
        # We can verify that we can access the list of operations in the graph
        # for op in graph.get_operations():
        #     print(op.name)

        # We access the input and output nodes
        x = graph.get_tensor_by_name('prefix/x:0')
        y = graph.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')
        # We launch a Session
        with tf.Session(graph=graph) as sess:
            # Additionally img is converted to greyscale
            observation = fun_img_preprocessing(observation, 48, 96)
            # this outputs omega, the desired angular velocity
            action = sess.run(y, feed_dict={
                x: observation
            })
            action = [action[0, 1], action[0, 0]]

            return action

    def on_received_get_commands(self, context: Context):
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

def simulation_scaling(action):
    """
    Remark: Use this if not mapping to PWM directly
    Scales the action based on internal simulation parameters (equivalent to standard Duckiebot parameters)
    :param action: left, right wheel commands based on imitation learning from logs
    :return: scaled action
    """
    # Distance between the wheels
    baseline = 0.102
    # assuming same motor constants k for both motors
    k_r = 27.0
    k_l = 27.0
    gain = 1.0
    trim = 0.0
    radius = 0.0318

    original_v = (action[0] + action[1])/2.0
    original_omega = (action[1] - action[0]) * radius / baseline / 2.0
    vel = 0.25
    omega = original_omega * vel / original_v

    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    omega_r = (vel + 0.5 * omega * baseline) / radius
    omega_l = (vel - 0.5 * omega * baseline) / radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv
    return [u_l, u_r]


def main():
    node = ImitationAgent()
    protocol = protocol_agent_duckiebot1
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
