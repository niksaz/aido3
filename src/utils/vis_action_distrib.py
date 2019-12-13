# Author: Mikita Sazanovich

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from src.learning.dataset import ConcreteTrainingDataset
from src.utils.config import CFG

PTS_LIMIT = 1000


def main():
    plt.figure(figsize=(8, 8))

    dataset_names = ['jetbrains', 'duckietown', 'loop_empty', 'udem1']
    colors = cm.rainbow(np.linspace(0, 1, len(dataset_names)))
    for dataset_name, color in zip(dataset_names, colors):
        dataset = ConcreteTrainingDataset(os.path.join('data', dataset_name), CFG.train_data_ratio, CFG.seed)
        print(f"{dataset_name} dataset's len is {len(dataset)}")

        points = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            points.append(y)
        points = np.array(points)

        if len(points) > PTS_LIMIT:
            np.random.seed(34)
            indexes = np.random.choice(len(points), PTS_LIMIT, replace=False)
            points = points[indexes]

        plt.scatter(points[:, 0], points[:, 1], s=1, color=color)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('left wheel')
    plt.ylabel('right wheel')
    plt.savefig('actions.png')


if __name__ == '__main__':
    main()
