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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

# Local imports
from pd_controller import PDController
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC
from utils import to_numpy, transform_images
from vint_utils import load_model
from vint_train.training.train_utils import get_action


class TopomapNavigationController:
    """Navigation controller using topological maps."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.robot_config = self._load_config("../config/robot.yaml")
        self.model_configs = self._load_config("../config/models.yaml")

        self.max_v = self.robot_config["max_v"]
        self.max_w = self.robot_config["max_w"]
        self.rate = self.robot_config["frame_rate"]

        self.controller = PDController()
        self.model = None
        self.noise_scheduler = None
        self.context_queue = []
        self.context_size = None

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
        """Load and configure the diffusion model."""

        model_config_path = self.model_configs[self.args.model]["config_path"]
        model_params = self._load_config(model_config_path)
        self.context_size = model_params["context_size"]

        ckpt_path = self.model_configs[self.args.model]["ckpt_path"]

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

        print(f"Loading model from {ckpt_path}")
        self.model = load_model(ckpt_path, model_params, self.device)
        self.model.eval()

        if model_params["model_type"] == "nomad":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

        self.model_params = model_params

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


    def _predict_actions_nomad(self) -> np.ndarray:
        """Predict actions using NoMaD model."""
        # Transform observation images
        obs_images = transform_images(
            self.context_queue,
            self.model_params["image_size"],
            center_crop=False
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(self.device)

        mask = torch.zeros(1).long().to(self.device)

        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)

        goal_images = []
        for g_img in self.topomap[start:end + 1]:
            goal_img = transform_images(
                g_img,
                self.model_params["image_size"],
                center_crop=False
            ).to(self.device)
            goal_images.append(goal_img)

        goal_images = torch.cat(goal_images, dim=0)

        obsgoal_cond = self.model(
            'vision_encoder',
            obs_img=obs_images.repeat(len(goal_images), 1, 1, 1),
            goal_img=goal_images,
            input_goal_mask=mask.repeat(len(goal_images))
        )

        dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dists = to_numpy(dists.flatten())

        min_idx = np.argmin(dists)
        self.closest_node = min_idx + start
        print(f"Closest node: {self.closest_node}, distance: {dists[min_idx]:.3f}")

        sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

        with torch.no_grad():
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

            noisy_action = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                device=self.device
            )

            num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler.set_timesteps(num_diffusion_iters)

            start_time = time.time()
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    'noise_pred_net',
                    sample=noisy_action,
                    timestep=k,
                    global_cond=obs_cond
                )

                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action
                ).prev_sample

            inference_time = time.time() - start_time
            print(f"Diffusion inference time: {inference_time:.3f}s")

        return to_numpy(get_action(noisy_action))

    def _predict_actions_vint(self) -> np.ndarray:
        """Predict actions using ViNT model."""
        # Get local topomap nodes within radius
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)

        distances = []
        waypoints = []
        batch_obs_imgs = []
        batch_goal_data = []

        for sg_img in self.topomap[start:end + 1]:
            transf_obs_img = transform_images(
                self.context_queue,
                self.model_params["image_size"]
            )
            goal_data = transform_images(
                sg_img,
                self.model_params["image_size"]
            )
            batch_obs_imgs.append(transf_obs_img)
            batch_goal_data.append(goal_data)

        # Predict distances and waypoints
        batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(self.device)
        batch_goal_data = torch.cat(batch_goal_data, dim=0).to(self.device)

        distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
        distances = to_numpy(distances)
        waypoints = to_numpy(waypoints)

        min_dist_idx = np.argmin(distances)

        # Select subgoal and waypoint
        if distances[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
            self.closest_node = start + min_dist_idx
        else:
            chosen_waypoint = waypoints[min(min_dist_idx + 1, len(waypoints) - 1)][self.args.waypoint]
            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)

        print(f"Closest node: {self.closest_node}, distance: {distances[min_dist_idx]:.3f}")

        return np.array([[chosen_waypoint]])

    def _get_base_velocity_command(self, waypoint: np.ndarray) -> List[float]:
        """Convert waypoint to base velocity command."""
        if self.model_params["normalize"]:
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

                    if self.model_params["model_type"] == "nomad":
                        predicted_actions = self._predict_actions_nomad()
                        chosen_waypoint = predicted_actions[0][self.args.waypoint]
                    else:
                        # Assume ViNT
                        predicted_actions = self._predict_actions_vint()
                        chosen_waypoint = predicted_actions[0][0]

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
        default="top1",
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