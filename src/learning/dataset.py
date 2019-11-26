# Author: Mikita Sazanovich

import os
import pickle
from typing import List

import cv2
import numpy as np

from src.utils.config import CFG
from src.utils.preprocessing import prepare_for_the_model


class Dataset:

    def __init__(self, data_dir):
        meta_path = os.path.join(data_dir, 'meta.pk')
        with open(meta_path, 'rb') as finput:
            boundaries = pickle.load(finput)
        index = []
        for i_start, i_end in boundaries:
            for k in range(i_end - i_start):
                inputs = []
                shift = 0
                input_id = max(i_start, i_start + k + shift)
                inputs.append(input_id)

                index.append(inputs)
        self.data_dir = data_dir
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        ids = self.index[item]
        for i in ids:
            img_filename = os.path.join(self.data_dir, f'{i}.png')
            img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
            img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = prepare_for_the_model(img_input)

        actions_filename = os.path.join(self.data_dir, f'{ids[0]}.npy')
        actions = np.load(actions_filename).astype(np.float32)
        return img_input, actions


class CombinedDataset:

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

    def __len__(self) -> int:
        total_len = 0
        for dataset in self.datasets:
            total_len += len(dataset)
        return total_len

    def __getitem__(self, item: int):
        for dataset in self.datasets:
            if item < len(dataset):
                return dataset[item]
            item -= len(dataset)
        return None
