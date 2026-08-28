#!/usr/bin/env python3

import argparse
import os
import time
from typing import List
import rclpy
from rclpy.node import Node
import numpy as np
import time

# import torch
# import torch.nn as nn
import yaml
from PIL import Image as PILImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray, Int32, Float32
# import onnxruntime as ort

# from utils import pil_to_numpy_array
import jax
import numpy as np
from crossformer.model.crossformer_model import CrossFormerModel
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

# UTILS
from topic_names import (
    IMAGE_TOPIC,
    WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    CLOSEST_NODE_TOPIC,
)

# CONSTANT
WORK_DIR = "/workspace/src/visualnav-transformer/deployment/" # ALWAYS DEPLOY INSIDE DOCKER
ROBOT_CONFIG_PATH =f"{WORK_DIR}config/robot.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
RATE = ROBOT_CONF["frame_rate"] 


from utils_onnx import msg_to_pil, transform_images, load_model_onnx

def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_numpy_array(image_input, target_size: tuple = (224, 224)) -> np.ndarray:
    """Convert PIL image or numpy array to numpy array with proper formatting for Crossformer."""

    if isinstance(image_input, PILImage.Image):

        if image_input.size != target_size:
            image_input = image_input.resize(target_size)
        img_array = np.array(image_input)
    elif isinstance(image_input, np.ndarray):

        img_array = image_input.copy()

        if img_array.shape[:2] != target_size:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))
            elif len(img_array.shape) == 2:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8), mode="L")
            else:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))

            pil_temp = pil_temp.resize(target_size)
            img_array = np.array(pil_temp)
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    return img_array


class TopomapNavigationController(Node):
    """Navigation controller using topological maps."""

    def __init__(self, args: argparse.Namespace):
        super().__init__('topomap_navigation_controller')
        self.args = args
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.robot_config = self._load_config("../config/robot.yaml")
        # self.model_configs = self._load_config("../config/models.yaml")
        self.max_v = self.robot_config["max_v"]
        self.max_w = self.robot_config["max_w"]
        self.rate = self.robot_config["frame_rate"]
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Create subscription
        # print(IMAGE_TOPIC)
        self.last_img_time = 0
        self.subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,  # Replace with your actual topic name
            self._callback_obs_ctrl_rate,
            qos_profile
        )
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )


        ######    Publisher used for data collection #############################

        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 1)
        self.goal_img_pub = self.create_publisher(Image, "/topoplan/goal_img", 1)
        self.subgoal_img_pub = self.create_publisher(Image, "/topoplan/subgoal_img", 1)
        self.closest_node_img_pub = self.create_publisher(Image, "/topoplan/closest_node_img", 1)
        self.closest_node_pub = self.create_publisher(Int32, CLOSEST_NODE_TOPIC, 10)
        self.distances_pub = self.create_publisher(Float32MultiArray, "/distances", 1)
        self.inference_pub = self.create_publisher(Float32, "/inference_time", 10)


        #########################################################################


        self.create_timer(1.0 / self.rate, self.run)

        self.model = None
        self.task = None
        self.noise_scheduler = None
        self.context_queue = []
        self.context_size = 5
        self.normalize = True
        self.rng_key = jax.random.PRNGKey(42)

        self.dist_pred_network, self.dist_model_params = self._load_dist_predictor()

        self.robot_model = self.args.robot_model

        self.closest_node = 0

        self.reached_goal = False

        self._setup_model()
        self._load_topomap()

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


    def _setup_model(self):
        # self.model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")
        self.model = CrossFormerModel.load_pretrained("/workspace/src/visualnav-transformer/deployment/model_weights/crossformer")

        # print("loaded crossformer")
        # self.model = CrossFormerModel.load_pretrained(
        #     "/root/.cache/huggingface/hub/models--rail-berkeley--crossformer/snapshots/c7dea2691aed3656537c5126a0a77df84a28abd7"
        # )
        self.unnormalization_statistics = dict(
            (stat_name, stat_value[:4, ...])
            for (stat_name, stat_value) in self.model.dataset_statistics[
                "omnimimic_gnm_dataset"
            ]["action"].items()
        )
        # self.model.save_pretrained(9999999999, "/workspace/src/visualnav-transformer/deployment/model_weights/crossformer")

    def _load_topomap(self):
        """Load topological map images."""
        topomap_dir = f"../topomaps/images/{self.args.dir}"
        if not os.path.exists(topomap_dir):
            raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")

        topomap_filenames = sorted(
            os.listdir(topomap_dir), key=lambda x: int(x.split(".")[0])
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

        # print(f"Goal node: {self.goal_node}")


    # The authors of crossformer used vint for distance prediction
    def _load_dist_predictor(self):
        model_params = {"normalize": True, "context_size": 5, "image_size": [85, 64]}
        model = load_model_onnx("vint")

        return model, model_params

    # This make it to match as control frequency
    def _callback_obs_ctrl_rate(self, msg: Image):
        
        current_time = time.time()
        if current_time - self.last_img_time < 1.0 / RATE:
            return  # Skip this frame
        
        self.last_img_time = current_time
        
        obs_img = msg_to_pil(msg)
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def _image_cb(self, msg: Image):
        # self.get_logger().info(f'Received image: {msg.width}x{msg.height}')
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(msg_to_pil(msg))
        else:
            self.context_queue.pop(0)
            self.context_queue.append(msg_to_pil(msg))

    def _predict_actions(self) -> np.ndarray:

        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        # import pdb; pdb.set_trace()
        crop = True
        
        start_time = time.time()
        # Transform observation once
        transf_obs_img = transform_images(
            list(self.context_queue), self.dist_model_params["image_size"], center_crop=crop
        )

        # Vectorized goal processing
        goal_imgs = self.topomap[start : end + 1]
        batch_goal_data_np = np.concatenate(
            [
                transform_images(
                    sg_img, self.dist_model_params["image_size"], center_crop=crop
                )
                for sg_img in goal_imgs
            ],
            axis=0,
        ).astype("float32")

        # Repeat observation for batch
        num_goals = len(goal_imgs)
        batch_obs_imgs_np = np.tile(transf_obs_img, (num_goals, 1, 1, 1)).astype(
            "float32"
        )

        ort_inputs = {
            "obs_img": batch_obs_imgs_np,
            "goal_img": batch_goal_data_np,
        }
        distances = self.dist_pred_network.run(None, ort_inputs)[0]
        # import pdb; pdb.set_trace()
        # print(f"Inference time without torch {time.time() - time_0}")

        min_dist_idx = np.argmin(distances)
        self.closest_node = start + min_dist_idx

        closest_node_msg = Int32()
        closest_node_msg.data = int(self.closest_node)
        self.closest_node_pub.publish(closest_node_msg)


        if distances[min_dist_idx] > self.args.close_threshold:
            sg_idx = self.closest_node
        else:
            sg_idx = min(self.closest_node + 1, self.goal_node)
        sg_idx = min(self.closest_node + 1, self.goal_node)

        print("Closest node", self.closest_node)


        target_goal_image = self.topomap[self.goal_node] #self.topomap[sg_idx]

        goal_img_np = pil_to_numpy_array(target_goal_image, target_size=(224, 224))

        goal_img_np = goal_img_np[None, ...]
        task = self.model.create_tasks(goals={"image_nav": goal_img_np})
        # print("after task")

        observation = self._prepare_crossformer_observation()
        self.rng_key, subkey = jax.random.split(self.rng_key)
        # print("after observation")

        action = self.model.sample_actions(
            observation,
            task,
            head_name="nav",
            rng=subkey,
            unnormalization_statistics=self.unnormalization_statistics,
        )
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f} seconds")

        inference_time_msg = Float32()
        inference_time_msg.data = inference_time
        self.inference_pub.publish(inference_time_msg)

        action = np.array(action, dtype=np.float64)

        return action

    def _prepare_crossformer_observation(self) -> dict:
        img_stack, timestep_mask = self._stack_and_pad(self.context_queue, max_length=5)

        observation = {"image_nav": img_stack, "timestep_pad_mask": timestep_mask}

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

    def run(self):
        """Main navigation loop."""
        # print("Starting topological navigation...")
        # print(f"Goal: reach node {self.goal_node}")

        try:
            # while not self.reached_goal:

                if len(self.context_queue) > self.context_size:
                    # print("Predicting action...")
                    chosen_waypoint = np.zeros(4)

                    predicted_actions = self._predict_actions()
                    chosen_waypoint = predicted_actions[0][0]
                    chosen_waypoint = np.array(chosen_waypoint, dtype=np.float64)
                    if len(chosen_waypoint) == 2:
                        chosen_waypoint = np.pad(chosen_waypoint, (0, 2), "constant")


                    waypoint_msg = Float32MultiArray()
                    waypoint_msg.data = chosen_waypoint.tolist()
                    self.waypoint_pub.publish(waypoint_msg)

                print(f"Closest node: {self.closest_node}")
                self.reached_goal = bool(self.closest_node == self.goal_node)
                self.goal_pub.publish(Bool(data=self.reached_goal))
                if self.reached_goal:
                    print("Goal reached!")
                    # break


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
    # parser.add_argument(
    #     "--model", "-m",
    #     default="nomad",
    #     type=str,
    #     help="Model name (check ../config/models.yaml) (default: nomad)"
    # )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="Index of waypoint for navigation (default: 2)",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="mist_office_new_chair",
        type=str,
        help="Path to topomap images directory (default: topomap)",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="Goal node index (-1 for last node) (default: -1)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=0.5,
        type=int,
        help="Distance threshold for node localization (default: 3)",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=2,
        type=int,
        help="Number of local nodes to consider (default: 2)",
    )

    parser.add_argument(
        "--robot-model", "-rb", default="bunker", type=str, help="bunker"
    )
    args = parser.parse_args()

    rclpy.init()
    print("Node starting...")
    tnc = TopomapNavigationController(args)
    print("Starting spin...")
    try:
        rclpy.spin(tnc)
    except KeyboardInterrupt:
        pass
    finally:
        tnc.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
