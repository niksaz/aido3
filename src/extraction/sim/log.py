# Author: Mikita Sazanovich

import argparse
import os
import pickle
import shutil

import cv2
import h5py
import numpy as np

from src.extraction.sim.env import launch_env
from src.extraction.sim.helpers import SteeringToWheelVelWrapper
from src.extraction.sim.teacher import PurePursuitExpert
from src.utils.preprocessing import preprocess_image

# Log configuration, you can pick your own values here
# the more the better? or the smarter the better?
EPISODES = 256
STEPS = 256

DEBUG = False


def save_dataset_as_h5(samples, dataset_dir, dataset_filename):
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_name = os.path.join(dataset_dir, dataset_filename)
    # check if file already exist in the data directory and if yes it is removed before saving the new file
    if os.path.isfile(dataset_name):
        os.remove(dataset_name)

    observations = np.array([sample[0] for sample in samples])
    actions = np.array([sample[1] for sample in samples])
    rewards = np.array([sample[2] for sample in samples])
    dones = np.array([sample[3] for sample in samples])
    print(observations.dtype, observations.shape)
    print(actions.dtype, actions.shape)
    print(rewards.dtype, rewards.shape)
    print(dones.dtype, dones.shape)

    print(f'Saving {len(samples)} samples to {dataset_name}')

    f = h5py.File(dataset_name, 'w')
    variant = f.create_group('split')
    group = variant.create_group('mix')
    group.create_dataset(name='observation', data=observations, compression='gzip')
    group.create_dataset(name='action', data=actions, compression='gzip')
    group.create_dataset(name='reward', data=rewards, compression='gzip')
    group.create_dataset(name='done', data=dones, compression='gzip')


def save_dataset_as_files(samples, boundaries, dataset_dir):
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    for i, sample in enumerate(samples):
        sample_filename = os.path.join(dataset_dir, f'{i}.png')
        cv2.imwrite(sample_filename, sample[0])
        action_filename = os.path.join(dataset_dir, f'{i}.npy')
        action = sample[1].astype(np.float32)
        np.save(action_filename, action)

    meta_filename = os.path.join(dataset_dir, 'meta.pk')
    with open(meta_filename, 'wb') as file_out:
        pickle.dump(boundaries, file_out, protocol=2)


def check_whether_simulator_invalid(info):
    return ('Simulator' in info
            and info['Simulator'].get('msg') == 'Stopping the simulator because we are at an invalid pose.')


def generate_samples_on(map_name, samples, boundaries):
    env = launch_env(map_name=map_name)

    # To convert to wheel velocities
    wrapper = SteeringToWheelVelWrapper()

    # this is an imperfect demonstrator... I'm sure you can construct a better one.
    expert = PurePursuitExpert(env=env)

    # let's collect our samples
    for episode in range(EPISODES):
        episode_samples = []
        for steps in range(STEPS):
            # we use our 'expert' to predict the next action.
            action = expert.predict(None)
            # Convert to wheel velocities
            action = wrapper.convert(action)
            observation, reward, done, info = env.step(action)
            if check_whether_simulator_invalid(info):
                done = True
            if done:
                break

            observation = preprocess_image(observation, cv2.COLOR_BGR2RGB)
            episode_samples.append([observation, action, reward, done, info])

            if DEBUG:
                env.render()
        env.reset()

        if len(episode_samples) != STEPS:
            print('Not including the episode since it has been unsuccessful...')
        else:
            samples_start = len(samples)
            samples.extend(episode_samples)
            samples_end = len(samples)
            boundaries.append([samples_start, samples_end])

        print(f'Finished {episode+1}/{EPISODES} episodes of {map_name}.'
              f'Total samples: {len(samples)}/{(episode+1)*STEPS}')

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('map_name', type=str, help='The name of the map to gather logs on')  # loop_empty, udem1
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    samples = []
    boundaries = []

    generate_samples_on(args.map_name, samples, boundaries)

    save_dataset_as_files(samples, boundaries, os.path.join('data', args.map_name))


if __name__ == '__main__':
    main()
