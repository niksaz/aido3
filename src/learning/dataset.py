# Author: Mikita Sazanovich

import os
import pickle
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np

from src.utils.config import CFG
from src.utils.preprocessing import prepare_for_the_model


class Dataset(ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: int):
        pass


class TrainingDataset(Dataset):

    @abstractmethod
    def get_train_indices(self):
        pass

    @abstractmethod
    def get_test_indices(self):
        pass


class ConcreteDataset(Dataset):

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
        self.__data_dir = data_dir
        self.__index = index[::CFG.dataset_subsample]
        self.__data = []
        for i in range(len(self.__index)):
            self.__data.append(self.__load_data(i))

    def __load_data(self, item: int):
        ids = self.__index[item]
        for i in ids:
            img_filename = os.path.join(self.__data_dir, f'{i}.png')
            img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
            img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = prepare_for_the_model(img_input)

        actions_filename = os.path.join(self.__data_dir, f'{ids[0]}.npy')
        actions = np.load(actions_filename).astype(np.float32)
        return img_input, actions

    def __len__(self) -> int:
        return len(self.__index)

    def __getitem__(self, item: int):
        return self.__data[item]


class ConcreteTrainingDataset(ConcreteDataset, TrainingDataset):

    def __init__(self, data_dir: str, train_data_ratio: float, seed: int):
        super().__init__(data_dir)
        random_state = np.random.RandomState(seed=seed)
        indices = np.arange(self.__len__())
        random_state.shuffle(indices)
        train_index_limit = int(len(indices) * train_data_ratio)
        self.train_indices = indices[:train_index_limit]
        self.test_indices = indices[train_index_limit:]

    def get_train_indices(self):
        return self.train_indices

    def get_test_indices(self):
        return self.test_indices


class CombinedTrainingDataset(TrainingDataset):

    def __init__(self, datasets: List[ConcreteTrainingDataset]):
        self.__datasets = datasets
        train_indices_list = []
        test_indices_list = []
        for k, dataset in enumerate(datasets):
            for internal_i in dataset.get_train_indices():
                train_indices_list.append(self.__internal_to_external(internal_i, k))
            for internal_i in dataset.get_test_indices():
                test_indices_list.append(self.__internal_to_external(internal_i, k))
        self.__train_indices = np.array(train_indices_list, dtype=np.int)
        self.__test_indices = np.array(test_indices_list, dtype=np.int)

    def __len__(self) -> int:
        res = sum(map(lambda dataset: len(dataset), self.__datasets))
        return res

    def __getitem__(self, item: int):
        data_pos, internal_i = self.__external_to_internal(item)
        return self.__datasets[data_pos][internal_i]

    def get_train_indices(self):
        return self.__train_indices

    def get_test_indices(self):
        return self.__test_indices

    def get_datasets(self):
        return self.__datasets

    def __internal_to_external(self, internal_i, data_pos):
        external_i = internal_i
        external_i += sum(map(lambda dataset: len(dataset), self.__datasets[:data_pos]))
        return external_i

    def __external_to_internal(self, external_i):
        internal_i = external_i
        for data_pos, dataset in enumerate(self.__datasets):
            if internal_i < len(dataset):
                return data_pos, internal_i
            else:
                internal_i -= len(dataset)
        raise IndexError(f'__external_to_internal for {external_i} is undefined')
