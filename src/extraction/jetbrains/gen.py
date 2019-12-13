# Author: Mikita Sazanovich

import os
import glob
import csv
import numpy as np
import pickle
import shutil
import cv2
import argparse
from src.utils.preprocessing import preprocess_image


def str_list_to_numpy_float64(slist):
    return np.array(slist).astype(np.float64)


def sync_cmd_with_images(wheels, image_timestamps):
    indices = []
    # for each omega velocity, find the respective image
    for cmd_index, (cmd_time, left_vel, right_vel) in enumerate(wheels):
        # we keep only the data for which the duckiebot is moving (we do not want the duckiebot to learn to remain at rest)
        if (left_vel != 0) & (right_vel != 0):

            # find index of image with the closest timestamp to wheels' velocities timestamp
            img_index = (np.abs(image_timestamps - cmd_time)).argmin()

            # The image precedes the omega velocity, thus image's timestamp must be smaller
            if ((image_timestamps[img_index] - cmd_time) > 0) & (img_index - 1 < 0):

                # if the image appears after the velocity and there is no previous image, then
                # there is no safe synchronization and the data should not be included
                continue
            else:
                # if the image appears after the velocity, in this case we know that there is previous image and we
                # should prefer it
                if (image_timestamps[img_index] - cmd_time) > 0:
                    img_index = img_index - 1

                indices.append((cmd_index, img_index))

    print(f'Total wheels: {len(wheels)}, Total images: {len(image_timestamps)}, Synched: {len(indices)}')
    return indices


def generate_data_pairs(indices, wheels, image_paths):
    samples = []
    actions = []
    for i, (cmd_index, img_index) in enumerate(indices):
        sample = cv2.imread(image_paths[img_index], cv2.IMREAD_COLOR)
        sample = preprocess_image(sample, cv2.COLOR_BGR2RGB)
        samples.append(sample)

        wheel_cmd = wheels[cmd_index]
        action = np.array([wheel_cmd[1], wheel_cmd[2]], dtype=np.float32)
        actions.append(action)
    return samples, actions


def save_dataset_as_files(samples, actions, boundaries, dataset_dir):
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    for i, (sample, action) in enumerate(zip(samples, actions)):
        sample_filename = os.path.join(dataset_dir, f'{i}.png')
        cv2.imwrite(sample_filename, cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        action_filename = os.path.join(dataset_dir, f'{i}.npy')
        np.save(action_filename, action)

    meta_filename = os.path.join(dataset_dir, 'meta.pk')
    with open(meta_filename, 'wb') as file_out:
        pickle.dump(boundaries, file_out, protocol=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/niksaz/Downloads/dec7_orig')
    parser.add_argument('--out_dir', type=str, default='/Users/niksaz/JBR/aido3/data/jetbrains')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    raw_data_dir = args.data_dir
    runs_names = os.listdir(raw_data_dir)

    boundaries = []
    samples = []
    actions = []

    for run_name in runs_names:
        run_path = os.path.join(raw_data_dir, run_name)
        wheels_filename = 'wheels_cmd_topic.csv'
        wheels_path = os.path.join(run_path, wheels_filename)
        images = glob.glob(f'{run_path}/*.jpg')
        images = sorted(images)

        images_timestamps = []
        for image_path in images:
            timestamp = os.path.basename(image_path).rsplit('.', 1)[0]
            images_timestamps.append(timestamp)
        images_timestamps = str_list_to_numpy_float64(images_timestamps)

        reader = csv.reader(open(wheels_path, 'r'), delimiter=',')
        wheels = []
        for row in reader:
            np_row = str_list_to_numpy_float64(row)
            wheels.append(np_row)
        wheels = np.array(wheels, dtype=np.float64)

        indices = sync_cmd_with_images(wheels, images_timestamps)

        run_samples, run_actions = generate_data_pairs(indices, wheels, images)

        run_start = len(samples)
        run_end = run_start + len(run_samples)
        boundaries.append([run_start, run_end])
        samples.extend(run_samples)
        actions.extend(run_actions)

    print(f'Runs: {len(boundaries)}, samples: {len(samples)}, actions: {len(actions)}')
    save_dataset_as_files(samples, actions, boundaries, args.out_dir)


if __name__ == '__main__':
    main()
