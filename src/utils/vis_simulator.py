# Author: Mikita Sazanovich

import pprint
import argparse

import matplotlib

matplotlib.use("macOSX")
print(matplotlib.get_backend())

import matplotlib.pyplot as plt
import numpy as np
from src.extraction.sim.env import launch_env
from gym_duckietown.simulator import get_dir_vec
from gym_duckietown.graphics import bezier_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('map_name', type=str, help='The name of the map to draw the map for')  # loop_empty, udem1
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = launch_env(map_name=args.map_name)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(env.map_data)

    points1 = []
    points2 = []

    for i in range(env.grid_width):
        for j in range(env.grid_height):
            # Get the tile type and angle
            tile = env._get_tile(i, j)
            if tile is None:
                continue

            # print(tile)
            kind = tile['kind']
            angle = tile['angle']
            color = tile['color']
            texture = tile['texture']
            print(i, j, kind, angle, color, texture)

            if tile['drivable']:
                # Find curve with largest dotproduct with heading
                curves = env._get_tile(i, j)['curves']
                curve_headings = curves[:, -1, :] - curves[:, 0, :]
                curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
                dirVec = get_dir_vec(angle)
                dot_prods = np.dot(curve_headings, dirVec)

                n = 5
                pts = env._get_curve(i, j)
                for idx, pt in enumerate(pts):
                    pts = [bezier_point(pt, i / (n - 1)) for i in range(n)]
                    if idx == np.argmax(dot_prods):
                        for p in pts:
                            points1.append(np.array(p))
                    else:
                        for p in pts:
                            points2.append(np.array(p))

    points1 = np.array(points1)
    points2 = np.array(points2)

    plt.scatter(points1[:, 2], points1[:, 0], s=1, c='r')
    plt.scatter(points2[:, 2], points2[:, 0], s=1, c='b')
    plt.show()


if __name__ == '__main__':
    main()
