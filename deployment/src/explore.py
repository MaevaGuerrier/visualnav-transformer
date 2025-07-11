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
from utils import to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action


class RobotNavigationController:
    """Navigation controller for robot using diffusion models."""

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

        self._setup_environment()
        self._setup_model()

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
            robot_model='locobot_wx250s',
            with_camera=True
        )

        obs, _ = self.env.reset()
        self.arm_joint_states = obs['state'][:6]

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

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.model_params = model_params

    def _update_context_queue(self, new_image):
        """Update the context queue with a new observation."""
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(new_image)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(new_image)

    def _predict_actions(self) -> np.ndarray:
        """Predict actions using the diffusion model."""
        obs_images = transform_images(
            self.context_queue,
            self.model_params["image_size"],
            center_crop=False
        ).to(self.device)

        fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(self.device)
        mask = torch.ones(1).long().to(self.device)

        with torch.no_grad():
            obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)

            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

            noisy_action = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                device=self.device
            )

            # Diffusion denoising process
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])

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
            print(f"Inference time: {inference_time:.3f}s")

        return to_numpy(get_action(noisy_action))

    def _get_base_velocity_command(self, waypoint: np.ndarray) -> List[float]:
        """Convert waypoint to base velocity command."""
        if self.model_params["normalize"]:
            waypoint *= (self.max_v / self.rate)
        return self.controller.get_velocity(waypoint)

    def run(self):
        """Main navigation loop."""
        print("Starting navigation loop...")

        try:
            while True:
                obs, _, _, _, _= self.env.step(list(self.arm_joint_states) + [0, 0])
                current_image = obs['camera']

                self._update_context_queue(current_image)

                chosen_waypoint = np.zeros(4)
                if len(self.context_queue) > self.context_size:
                    predicted_actions = self._predict_actions()
                    selected_action = predicted_actions[0]
                    chosen_waypoint = selected_action[self.args.waypoint]

                base_velocity_command = self._get_base_velocity_command(chosen_waypoint)

                action = list(self.arm_joint_states) + base_velocity_command
                print(f'Executing action: {action}')
                obs, _, _, _, _ = self.env.step(action)

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nNavigation stopped by user")
        except Exception as e:
            print(f"Error during navigation: {e}")
            raise


def main():
    """Main function to parse arguments and run navigation."""
    parser = argparse.ArgumentParser(
        description="Run diffusion-based navigation on the Locobot"
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
        help="Index of waypoint for navigation (0-4) (default: 2)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        default=8,
        type=int,
        help="Number of action samples from exploration model (default: 8)"
    )

    args = parser.parse_args()

    navigator = RobotNavigationController(args)
    navigator.run()


if __name__ == "__main__":
    main()