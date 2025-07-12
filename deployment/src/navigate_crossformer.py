#!/usr/bin/env python3

import argparse
import os
import time
from typing import List

import gymnasium as gym
import numpy as np
import robo_gym
import torch
import torch.nn as nn
import yaml
from PIL import Image as PILImage

from pd_controller import PDController
from utils import pil_to_numpy_array
import jax
import numpy as np
from crossformer.model.crossformer_model import CrossFormerModel


class TopomapNavigationController:
    """Navigation controller using topological maps."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.robot_config = self._load_config("../config/robot.yaml")
        # self.model_configs = self._load_config("../config/models.yaml")

        self.max_v = self.robot_config["max_v"]
        self.max_w = self.robot_config["max_w"]
        self.rate = self.robot_config["frame_rate"]

        self.controller = PDController()
        self.model = None
        self.task = None
        self.noise_scheduler = None
        self.context_queue = []
        self.context_size = 3
        self.normalize = True
        self.rng_key = jax.random.PRNGKey(42)

        self.robot_model = self.args.locobot_model

        if self.robot_model == 'locobot_wx250s':
            self.dof = 6
        elif self.robot_model == 'locobot_px100':
            self.dof = 4
        else:
            self.dof = 5

        self.closest_node = 0
        self.reached_goal = False

        self._setup_environment()
        self._setup_model()
        self._load_topomap()

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_environment(self):
        """Initialize the robo-gym environment."""
        self.env = gym.make(
            'EmptyEnvironmentInterbotixRRob-v0',
            rs_address='127.0.0.1:50051',
            gui=True,
            robot_model=self.robot_model,
            with_camera=True
        )

        obs, _ = self.env.reset()
        self.arm_joint_states = obs['state'][:self.dof]

    def _setup_model(self):

        self.model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")


    def _load_topomap(self):
        """Load topological map images."""
        topomap_dir = f"../topomaps/images/{self.args.dir}"
        if not os.path.exists(topomap_dir):
            raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")

        topomap_filenames = sorted(
            os.listdir(topomap_dir),
            key=lambda x: int(x.split(".")[0])
        )

        self.topomap = []
        for filename in topomap_filenames:
            image_path = os.path.join(topomap_dir, filename)
            self.topomap.append(PILImage.open(image_path))

        print(f"Loaded topomap with {len(self.topomap)} nodes")

        if self.args.goal_node == -1:
            self.goal_node = len(self.topomap) - 1
        else:
            assert 0 <= self.args.goal_node < len(self.topomap), "Invalid goal index"
            self.goal_node = self.args.goal_node

        print(f"Goal node: {self.goal_node}")

    def _update_context_queue(self, new_image):
        """Update the context queue with a new observation."""
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(new_image)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(new_image)


    def _predict_actions(self) -> np.ndarray:

        goal_idx = min(self.closest_node + 1, self.goal_node)
        target_goal_image = self.topomap[goal_idx]

        goal_img_np = pil_to_numpy_array(target_goal_image, target_size=(224, 224))

        goal_img_np = goal_img_np[None, ...]
        task = self.model.create_tasks(
            goals={"image_nav": goal_img_np})

        observation = self._prepare_crossformer_observation()
        self.rng_key, subkey = jax.random.split(self.rng_key)

        action = self.model.sample_actions(observation, task, head_name="nav", rng=subkey)

        action = np.array(action, dtype=np.float64)

        print(f"Sampled action: {action}")

        if goal_idx > self.closest_node:
            self.closest_node = goal_idx

        return action

    def _prepare_crossformer_observation(self) -> dict:
        img_stack, timestep_mask = self._stack_and_pad(self.context_queue, max_length=5)

        observation = {
            "image_nav": img_stack,
            "timestep_pad_mask": timestep_mask
        }

        return observation

    def _stack_and_pad(self, images_list, max_length=5):
        """
        Stack and pad observations
        """
        np_images = []
        for img in images_list:
            img_array = pil_to_numpy_array(img)
            np_images.append(img_array)

        actual_length = len(np_images)

        if actual_length == 0:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            np_images = [dummy_img]
            actual_length = 1

        while len(np_images) < max_length:
            np_images.insert(0, np.zeros_like(np_images[0]))

        np_images = np_images[-max_length:]

        img_stack = np.stack(np_images, axis=0)
        img_stack = img_stack[None, ...]

        timestep_mask = np.zeros((1, max_length), dtype=bool)

        start_idx = max_length - min(actual_length, max_length)
        timestep_mask[0, start_idx:] = True

        return img_stack, timestep_mask

    def _get_base_velocity_command(self, waypoint: np.ndarray) -> List[float]:
        """Convert waypoint to base velocity command."""
        if self.normalize:
            waypoint[:2] *= (self.max_v / self.rate)
        return self.controller.get_velocity(waypoint)

    def run(self):
        """Main navigation loop."""
        print("Starting topological navigation...")
        print(f"Goal: reach node {self.goal_node}")

        try:
            while not self.reached_goal:
                obs, _, _, _, _= self.env.step(list(self.arm_joint_states) + [0, 0])
                current_image = obs['camera']

                self._update_context_queue(current_image)

                chosen_waypoint = np.zeros(4)

                if len(self.context_queue) > self.context_size:

                    predicted_actions = self._predict_actions()
                    chosen_waypoint = predicted_actions[0][0]
                    chosen_waypoint = np.array(chosen_waypoint, dtype=np.float64)
                    if len(chosen_waypoint) == 2:
                        chosen_waypoint = np.pad(chosen_waypoint, (0, 2), 'constant')

                base_velocity_command = self._get_base_velocity_command(chosen_waypoint)

                action = list(self.arm_joint_states) + base_velocity_command
                print(f'Executing action: {action}')
                obs, _, _, _, _ = self.env.step(action)

                self.reached_goal = (self.closest_node == self.goal_node)
                if self.reached_goal:
                    print("Goal reached!")
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nNavigation stopped by user")
        except Exception as e:
            print(f"Error during navigation: {e}")
            raise


def main():
    """Main function to parse arguments and run navigation."""
    parser = argparse.ArgumentParser(
        description="Run topological navigation on the Locobot"
    )
    parser.add_argument(
        "--model", "-m",
        default="nomad",
        type=str,
        help="Model name (check ../config/models.yaml) (default: nomad)"
    )
    parser.add_argument(
        "--waypoint", "-w",
        default=2,
        type=int,
        help="Index of waypoint for navigation (default: 2)"
    )
    parser.add_argument(
        "--dir", "-d",
        default="map2",
        type=str,
        help="Path to topomap images directory (default: topomap)"
    )
    parser.add_argument(
        "--goal-node", "-g",
        default=-1,
        type=int,
        help="Goal node index (-1 for last node) (default: -1)"
    )
    parser.add_argument(
        "--close-threshold", "-t",
        default=3,
        type=int,
        help="Distance threshold for node localization (default: 3)"
    )
    parser.add_argument(
        "--radius", "-r",
        default=4,
        type=int,
        help="Number of local nodes to consider (default: 4)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        default=8,
        type=int,
        help="Number of action samples for NoMaD (default: 8)"
    )
    parser.add_argument(
        "--locobot-model", "-l",
        default='locobot_wx250s',
        type=str,
        help="Locobot robot model"
    )
    args = parser.parse_args()

    navigator = TopomapNavigationController(args)
    navigator.run()


if __name__ == "__main__":
    main()