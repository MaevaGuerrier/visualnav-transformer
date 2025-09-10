import numpy as np
import yaml
from typing import Tuple

# from utils import clip_angle

CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)


def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


class PDController(object):
    def __init__(self):
        dt = 1 / robot_config["frame_rate"]
        self.dt = dt
        self.eps = 1e-8
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]

    def clip_angle(self, theta) -> float:
        """Clip angle to [-pi, pi]"""
        theta %= 2 * np.pi
        if -np.pi < theta < np.pi:
            return theta
        return theta - 2 * np.pi

    def get_velocity(self, waypoint: np.ndarray) -> Tuple[float]:
        """PD controller for the robot"""
        assert (
            len(waypoint) == 2 or len(waypoint) == 4
        ), "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < self.eps and np.abs(dy) < self.eps:
            v = 0
            w = clip_angle(np.arctan2(hy, hx)) / self.dt
        elif np.abs(dx) < self.eps:
            v = 0
            w = np.sign(dy) * np.pi / (2 * self.dt)
        else:
            v = dx / self.dt
            w = np.arctan(dy / dx) / self.dt
        v = np.clip(v, 0, self.max_v)
        w = np.clip(w, -self.max_w, self.max_w)
        return [v, w]
