# Author: Mikita Sazanovich

import gym
import gym_duckietown


def launch_env(map_name='loop_empty'):
    from gym_duckietown.simulator import Simulator
    env = Simulator(
        seed=123,  # random seed
        map_name=map_name,
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        distortion=True,
    )
    return env
