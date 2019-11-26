# Author: Mikita Sazanovich

import os
import shutil

import h5py
import numpy as np
import cv2
import pickle

from src.extraction.sim.env import launch_env
from src.extraction.sim.helpers import SteeringToWheelVelWrapper
from src.extraction.sim.teacher import PurePursuitExpert
from src.utils.preprocessing import preprocess_image

# Log configuration, you can pick your own values here
# the more the better? or the smarter the better?
EPISODES = 200
STEPS = 512

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
    dataset_dir = os.path.join(dataset_dir, 'sim')
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


def main():
    env = launch_env()

    # To convert to wheel velocities
    wrapper = SteeringToWheelVelWrapper()

    # this is an imperfect demonstrator... I'm sure you can construct a better one.
    expert = PurePursuitExpert(env=env)

    samples = []
    boundaries = []
    # let's collect our samples
    for episode in range(EPISODES):
        episode_samples = []
        for steps in range(STEPS):
            # we use our 'expert' to predict the next action.
            action = expert.predict(None)
            # Convert to wheel velocities
            action = wrapper.convert(action)
            observation, reward, done, info = env.step(action)
            closest_point, _ = env.closest_curve_point(env.cur_pos, env.cur_angle)
            if closest_point is None:
                done = True
                break

            observation = preprocess_image(observation, cv2.COLOR_BGR2RGB)
            episode_samples.append([observation, action, reward, done, info])

            if DEBUG:
                env.render()
        env.reset()

        samples_start = len(samples)
        samples.extend(episode_samples)
        samples_end = len(samples)
        boundaries.append([samples_start, samples_end])

        print(f'Finished {episode+1}/{EPISODES} episodes. Total samples: {len(samples)}')

    env.close()

    save_dataset_as_files(samples, boundaries, 'data')


if __name__ == '__main__':
    main()
