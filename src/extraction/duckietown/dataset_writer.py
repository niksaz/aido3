# Author: Mikita Sazanovich

import pickle
import os
import shutil

import cv2
import numpy as np


class FilesDatasetWriter:
  def __init__(self, dataset_dir):
    self.dataset_dir = dataset_dir
    self.samples_written = 0
    self.episode_boundaries = []
    self._recreate_dataset_dir()

  def _recreate_dataset_dir(self):
    if os.path.exists(self.dataset_dir):
      shutil.rmtree(self.dataset_dir)
    os.makedirs(self.dataset_dir)

  def save_episode(self, synch_imgs, synch_data):
    episode_starts = self.samples_written
    for image, log in zip(synch_imgs, synch_data):
      sample_filename = os.path.join(self.dataset_dir, '{}.png'.format(self.samples_written))
      cv2.imwrite(sample_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
      action_filename = os.path.join(self.dataset_dir, '{}.npy'.format(self.samples_written))
      action = np.array([log[2], log[3]], dtype=np.float32)
      np.save(action_filename, action)
      self.samples_written += 1
    episode_ends = self.samples_written
    self.episode_boundaries.append([episode_starts, episode_ends])

  def save_boundaries(self):
    boundaries_filename = os.path.join(self.dataset_dir, 'meta.pk')
    with open(boundaries_filename, 'wb') as file_out:
      pickle.dump(self.episode_boundaries, file_out, protocol=2)
