# Author: Mikita Sazanovich

import numpy as np
import scipy.stats as stats

# parameters for the pure pursuit controller
PEAK_VELOCITY = 1.0
GAIN = 8
FOLLOWING_DISTANCE = 0.5


class PurePursuitExpert:
    def __init__(self, env, following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations

    def predict(self, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        if closest_point is None:
            return 0.0, 0.0  # Should return done in the environment

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = GAIN * -dot
        velocity = (PurePursuitExpert.__get_speed_density_at(steering)
                    / PurePursuitExpert.__get_speed_density_at(0.0)
                    * PEAK_VELOCITY)
        return velocity, steering

    @staticmethod
    def __get_speed_density_at(x):
        return stats.norm.pdf(x, 0.0, 2.0)
