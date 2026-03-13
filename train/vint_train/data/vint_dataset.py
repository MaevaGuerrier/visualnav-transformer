import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2

from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class ViNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
        flip_aug: bool = False,
        image_aug: bool = False,
        image_aug_params: Dict[str, Any] = {},
        learn_metric_distance: bool = False,
        metric_distance_for_negatives: bool = False,
        fluctuate_actions: bool = False,
        action_fluctuation_amount: float = 0.2,
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
            learn_metric_distance (bool): Whether to learn the metric distance (in meters) between observation 
                                          and goal positions instead of temporal distance (in timesteps)
            metric_distance_for_negatives (bool): If True and learn_metric_distance=True, compute actual metric
                                                  distance for negative goals. If False, use max_dist_cat for negatives.
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.flip_aug = flip_aug
        self.image_aug = image_aug
        if self.image_aug:
            self.obs_aug_transform = A.Compose([
                A.ColorJitter(**image_aug_params.get("color_jitter", {})),
                A.GaussianBlur(**image_aug_params.get("gaussian_blur", {})),
                A.CoarseDropout(**image_aug_params.get("coarse_dropout", {})),
                ToTensorV2(),
            ], additional_targets={f"image{i}": "image" for i in range(1, context_size+1)})
            self.goal_aug_transform = A.Compose([
                A.ColorJitter(**image_aug_params.get("color_jitter", {})),
                A.GaussianBlur(**image_aug_params.get("gaussian_blur", {})),
                A.CoarseDropout(**image_aug_params.get("coarse_dropout", {})),
                ToTensorV2(),
            ])


        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type
        self.learn_metric_distance = learn_metric_distance
        self.metric_distance_for_negatives = metric_distance_for_negatives
        self.fluctuate_actions = fluctuate_actions
        self.action_fluctuation_amount = action_fluctuation_amount

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        # Get metric waypoint spacing for this dataset (used for loss scaling)
        self.metric_waypoint_spacing = self.data_config.get("metric_waypoint_spacing", 0.1)
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        #self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)
        # wait until forked to open lmdb
        self.cache_filename = cache_filename
        self._image_cache = None

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        if self.negative_mining:
            goal_offset = np.random.randint(0, max_goal_dist + 1)
            if goal_offset == 0:
                trajectory_name, goal_time = self._sample_negative()
                return trajectory_name, goal_time, True
            else:
                goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
                return trajectory_name, goal_time, False
        else:
            # No negative mining - always sample from the same trajectory
            goal_offset = np.random.randint(1, max_goal_dist + 1)
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        if self._image_cache is None:
            self._image_cache = lmdb.open(
                self.cache_filename, 
                readonly=True, 
                lock=False,
                meminit=False
            )

        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                if image_buffer is None:
                    raise TypeError(f"Image not found in cache: {image_path}")
                image_bytes = bytes(image_buffer)
            
            with io.BytesIO(image_bytes) as f:
                return img_path_to_data(f, self.image_size)
        except TypeError as e:
            print(f"Failed to load image {image_path}: {e}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos
    
    def _compute_action_history(self, traj_data, curr_time):
        """
        Compute the action history for the context observations.
        
        For each context observation at time t, compute the action (waypoint)
        from t to t+waypoint_spacing. This represents the action that was
        executed from that observation.
        
        Args:
            traj_data: Dictionary containing trajectory data (position, yaw)
            curr_time: Current timestep (the last context observation)
            
        Returns:
            action_history: numpy array of shape (context_size, num_action_params)
        """
        # Compute the time indices for context observations
        # context_times includes curr_time, so we need context_size steps before it
        context_times = list(
            range(
                curr_time - self.context_size * self.waypoint_spacing,
                curr_time,
                self.waypoint_spacing,
            )
        )
        
        action_history = []
        traj_len = len(traj_data["position"])
        
        for t in context_times:
            # Compute action from t to t+waypoint_spacing
            start_idx = t
            end_idx = min(t + self.waypoint_spacing, traj_len - 1)
            
            start_pos = traj_data["position"][start_idx]
            end_pos = traj_data["position"][end_idx]
            start_yaw = traj_data["yaw"][start_idx]
            end_yaw = traj_data["yaw"][end_idx]
            
            if len(start_yaw.shape) == 2:
                start_yaw = start_yaw.squeeze()
            if len(end_yaw.shape) == 2:
                end_yaw = end_yaw.squeeze()
            
            # Transform to local coordinates
            start_pos_local = np.array([0, 0])  # Origin in local frame
            end_pos_local = to_local_coords(end_pos, start_pos, start_yaw)
            
            if self.learn_angle:
                # Compute yaw difference
                yaw_diff = end_yaw - start_yaw
                action = np.concatenate([end_pos_local, [yaw_diff]])
            else:
                action = end_pos_local
            
            action_history.append(action)
        
        action_history = np.array(action_history, dtype=np.float32)
        
        # Normalize if needed (same as future actions)
        if self.normalize:
            action_history[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        
        assert action_history.shape == (self.context_size, self.num_action_params), \
            f"action_history shape {action_history.shape} != {(self.context_size, self.num_action_params)}"
        
        return action_history
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            for k in traj_data:
                traj_data[k] = traj_data[k].astype(np.float32)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _apply_image_augmentation(self, image: torch.Tensor, aug_params: Dict[str, Any]) -> torch.Tensor:
        pass

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing:
                obs_image (torch.Tensor): tensor of shape [(context_size+1)*3, H, W] containing context images
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                action_label (torch.Tensor): tensor of shape (len_traj_pred, num_action_params) containing future actions
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance from observation to goal
                goal_pos (torch.Tensor): tensor of shape (2,) containing goal position in local coords
                dataset_index (torch.Tensor): index of the dataset for multi-dataset training
                action_mask (torch.Tensor): tensor of shape (1,) indicating if action should be used for loss
                metric_waypoint_spacing (torch.Tensor): metric spacing between waypoints for this dataset
                action_history (torch.Tensor): tensor of shape (context_size, num_action_params) containing past actions
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions (future trajectory)
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # Compute action history for context observations
        action_history = self._compute_action_history(curr_traj_data, curr_time)

        if self.flip_aug:
            if np.random.rand() < 0.5:
                actions[:, 1] *= -1 # flip y (y is left/right)
                if self.learn_angle: # flip yaw
                    actions[:, 2] *= -1
                goal_pos[1] *= -1

                obs_image = obs_image.flip(-1)
                goal_image = goal_image.flip(-1)

        if self.image_aug:
            # apply image augmentation to obs image and goal image
            # obs images should be applied the same augmentation
            # I don't know why obs_image has context_size + 1, probably bugged code by author
            goal_image = self.goal_aug_transform(image=goal_image.permute(1, 2, 0).numpy())["image"]
            obs_image_transformed = self.obs_aug_transform(
                **{f"image{i if i > 0 else ''}": obs_image[i*3:(i+1)*3].permute(1, 2, 0).numpy() for i in range(self.context_size+1)}
            )
            obs_image = torch.cat([
                obs_image_transformed[f"image{i if i > 0 else ''}"] for i in range(self.context_size+1)
            ])

        if self.fluctuate_actions:
            assert self.normalize, "Action fluctuation should only be used when actions are normalized"
            scale = 1 + np.random.uniform(-self.action_fluctuation_amount, self.action_fluctuation_amount)
            actions[:, :2] *= scale
            action_history[:, :2] *= scale

        #self._save_images(obs_image, goal_image, i)
        
        # Compute distances
        if self.learn_metric_distance and (self.metric_distance_for_negatives or not goal_is_negative):
            # Calculate Euclidean distance between observation and goal positions (in meters)
            # For negatives: only compute metric distance if metric_distance_for_negatives=True
            curr_pos = curr_traj_data["position"][curr_time]
            goal_pos_metric = goal_traj_data["position"][min(goal_time, len(goal_traj_data["position"]) - 1)]
            distance = np.linalg.norm(curr_pos - goal_pos_metric)
        elif goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        # Convert action_history to tensor and apply sin/cos encoding if needed
        action_history_torch = torch.as_tensor(action_history, dtype=torch.float32)
        if self.learn_angle:
            action_history_torch = calculate_sin_cos(action_history_torch)
        
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        # Use float32 for metric distance, int64 for temporal distance
        distance_dtype = torch.float32 if self.learn_metric_distance else torch.int64
        
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=distance_dtype),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(self.metric_waypoint_spacing, dtype=torch.float32),
            action_history_torch,
        )

    def _save_images(self, obs_image: torch.Tensor, goal_image: torch.Tensor, index: int) -> None:
        """
        Save the observation and goal images to disk for debugging purposes
        """
        import torchvision.utils as vutils
        print('saving debug images...')
        os.makedirs("debug_images", exist_ok=True)
        print(obs_image.shape, goal_image.shape)
        vutils.save_image(
            obs_image[-3:],
            f"debug_images/obs_image_{index}.png",
            normalize=False,
        )
        vutils.save_image(
            goal_image,
            f"debug_images/goal_image_{index}.png",
            normalize=False,
        )