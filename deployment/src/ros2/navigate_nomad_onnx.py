from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, List
import onnxruntime as ort
import gc

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, Float32
import torch
import yaml

from src.utils_onnx import msg_to_pil, transform_images, load_model_onnx
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.training.train_utils import get_action
from src.utils import to_numpy

# UTILS
from src.topic_names import (
    IMAGE_TOPIC,
    WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    CLOSEST_NODE_TOPIC,
)

# CONSTANTS
WORK_DIR = "/workspace/src/visualnav-transformer/deployment/" # ALWAYS DEPLOY INSIDE DOCKER
TOPOMAP_IMAGES_DIR = f"{WORK_DIR}topomaps/images"
MODEL_WEIGHTS_PATH = f"{WORK_DIR}model_weights/"
ROBOT_CONFIG_PATH =f"{WORK_DIR}config/robot.yaml"
MODEL_CONFIG_PATH = f"{WORK_DIR}../train/config/"

with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE = ROBOT_CONF["frame_rate"]  # Hz

# CAMERA ===============================================================

INTRINSICS = np.array([[235.7444344725863, 2.2822917369575983, 320.3212422370101],
                            [0.0,               237.67070839912813,  232.78147845844464],
                            [0.0,               0.0,                 1.0]])

CAMERA_HEIGHT = 0.560
CAMERA_X_OFFSET = 0.200
# first row last val offset along x (e.g -0.600 --> 60 cm forward)
# before last row last offset along z (vertical height, e.g 0.042 --> 4.2 cm)        
EXTRINSICS = np.array([[0, 0, 1, -CAMERA_X_OFFSET], 
                                [-1, 0, 0, -0.000],
                                [0, -1, 0, -CAMERA_HEIGHT],
                                [0, 0, 0, 1]])
DIST_COEFF = np.array([[-0.053129475318406234],
                         [ 0.03335273788977895],
                         [-0.031760136310879046],
                         [ 0.008394411829175783]])  # shape (4, 1)
VIZ_IMAGE_SIZE_FISHEYE = (640, 480) # (640, 480) orig fisheye image size

# ======================================================================

class NavigationNode(Node):
    """Sub‑goal navigation with topomap + trajectory visualisation."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("navigation")
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.vis_encoder, self.dist_pred, self.noise_pred, self.noise_scheduler, self.model_params = self._load_model()

        self.context_size: int = self.model_params["context_size"]
        assert self.context_size != None

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]

        self.bridge = CvBridge()
        self.context_queue = []
        self.last_ctx_time = self.get_clock().now()
        self.ctx_dt = 0.25

        self.current_waypoint = np.zeros(2)
        self.obstacle_points = None

        self.top_view_size = (400, 400)
        self.proximity_threshold = 0.8
        self.top_view_resolution = self.top_view_size[0] / self.proximity_threshold
        self.top_view_sampling_step = 5
        self.safety_margin = 0.17
        self.DIM = (640, 480)

        # Topological map ----------------------------------------------------
        self.topomap, self.goal_node = self._load_topomap(self.args.dir, self.args.goal_node)
        self.closest_node = 0

        self.create_subscription(Image, IMAGE_TOPIC, self._callback_obs_ctrl_rate, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )
        
        self.viz_pub = self.create_publisher(Image, "navigation_viz", 1)
        self.subgoal_pub = self.create_publisher(Image, "navigation_subgoal", 1)
        self.goal_pub_img = self.create_publisher(Image, "navigation_goal", 1)


        ######    Publisher used for data collection #############################

        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 1)
        self.goal_img_pub = self.create_publisher(Image, "/topoplan/goal_img", 1)
        self.subgoal_img_pub = self.create_publisher(Image, "/topoplan/subgoal_img", 1)
        self.closest_node_img_pub = self.create_publisher(Image, "/topoplan/closest_node_img", 1)
        self.closest_node_pub = self.create_publisher(Int32, CLOSEST_NODE_TOPIC, 10)
        self.distances_pub = self.create_publisher(Float32MultiArray, "/distances", 1)
        self.inference_pub = self.create_publisher(Float32, "/inference_time", 10)


        #########################################################################

        self.create_timer(1.0 / RATE, self._timer_cb)
        self.get_logger().info("Navigation node initialised. Waiting for images…")

        self.get_logger().info("=" * 60)
        self.get_logger().info("NAVIGATION NODE PARAMETERS")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Image topic: {IMAGE_TOPIC}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("ROBOT CONFIGURATION:")
        self.get_logger().info(f"  - Max linear velocity: {MAX_V} m/s")
        self.get_logger().info(f"  - Max angular velocity: {MAX_W} rad/s")
        self.get_logger().info(f"  - Frame rate: {RATE} Hz")
        self.get_logger().info(f"  - Safety margin: {self.safety_margin} m")
        self.get_logger().info(f"  - Proximity threshold: {self.proximity_threshold} m")
        self.get_logger().info("-" * 60)
        self.get_logger().info("CAMERA CONFIGURATION:")
        self.get_logger().info(f"  - Image dimensions: {self.DIM}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("MODEL CONFIGURATION:")
        self.get_logger().info(f"  - Device: {self.device}")
        self.get_logger().info(f"  - Context size: {self.context_size}")
        self.get_logger().info(f"  - Context update interval: {self.ctx_dt} seconds")
        self.get_logger().info(f"  - Image size: {self.model_params['image_size']}")
        self.get_logger().info(
            f"  - Normalize: {self.model_params.get('normalize', False)}"
        )

        self.get_logger().info("-" * 60)
        self.get_logger().info("TOPOLOGICAL MAP CONFIGURATION:")
        self.get_logger().info(f"  - Topomap directory: {self.args.dir}")
        self.get_logger().info(f"  - Number of nodes: {len(self.topomap)}")
        self.get_logger().info(f"  - Goal node: {self.goal_node}")
        self.get_logger().info(f"  - Search radius: {self.args.radius}")
        self.get_logger().info(f"  - Close threshold: {self.args.close_threshold}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("OBSTACLE AVOIDANCE CONFIGURATION:")
        self.get_logger().info(f"  - Top view size: {self.top_view_size}")
        self.get_logger().info(
            f"  - Top view resolution: {self.top_view_resolution:.2f} pixels/m"
        )
        self.get_logger().info(
            f"  - Top view sampling step: {self.top_view_sampling_step} pixels"
        )
        self.get_logger().info("-" * 60)
        self.get_logger().info("ROS TOPICS:")
        self.get_logger().info(f"  - Subscribing to: {IMAGE_TOPIC}")
        self.get_logger().info(f"  - Publishing waypoints to: {WAYPOINT_TOPIC}")
        self.get_logger().info(
            f"  - Publishing sampled actions to: {SAMPLED_ACTIONS_TOPIC}"
        )
        self.get_logger().info(
            f"  - Publishing navigation visualization to: /navigation_viz"
        )
        self.get_logger().info(f"  - Publishing subgoal image to: /navigation_subgoal")
        self.get_logger().info(f"  - Publishing goal image to: /navigation_goal")
        self.get_logger().info(
            f"  - Publishing goal reached status to: /topoplan/reached_goal"
        )
        self.get_logger().info("-" * 60)
        self.get_logger().info("EXECUTION PARAMETERS:")
        self.get_logger().info(f"  - Waypoint index: {self.args.waypoint}")
        self.get_logger().info(f"  - Number of samples: {self.args.num_samples}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("VISUALIZATION PARAMETERS:")
        self.get_logger().info(f"  - Pixels per meter: 3.0")
        self.get_logger().info(f"  - Lateral scale: 1.0")
        self.get_logger().info(f"  - Horizontal scale: 4.0")
        self.get_logger().info(f"  - Robot symbol length: 10 pixels")
        self.get_logger().info("=" * 60)

    # Helper: topomap
    # ------------------------------------------------------------------

    def _load_topomap(
        self, dir_path: str, goal_node: int
    ) -> Tuple[List[PILImage.Image], int]:
        topomap_filenames = sorted(
            os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, dir_path)),
            key=lambda x: int(x.split(".")[0]),
        )
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{dir_path}"
        num_nodes = len(os.listdir(topomap_dir))
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            topomap.append(PILImage.open(image_path))

        assert -1 <= goal_node < len(topomap), "Invalid goal index for the topomap"
        if goal_node == -1:
            goal_node = len(topomap) - 1

        return topomap, goal_node

    def _image_cb(self, msg: Image):

        self.context_queue.append(msg_to_pil(msg))
        # self.last_ctx_time = now
        # self.get_logger().info(
        #     f"Image added to context queue ({len(self.context_queue)})"
        # )

    # This make it to be actually 4hz same as control frequency
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

    def _timer_cb(self):
        if len(self.context_queue) <= self.context_size:
            return

        self._timer_cb_nomad()

        if self.closest_node == self.goal_node:
            self.get_logger().info("Reached goal! Stopping...")

    def _load_model(self):
        model_config_path = f"{MODEL_CONFIG_PATH}nomad.yaml"
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        vis_encoder = load_model_onnx("nomad_vision_encoder")
        print("loaded vision encoder onnx model")
        dist_pred = load_model_onnx("nomad_dist_pred_net")
        print("loaded distance predictor onnx model")
        noise_pred = load_model_onnx("nomad_noise_pred_net")
        print("loaded noise predictor onnx model")


        return vis_encoder, dist_pred, noise_pred, noise_scheduler, model_params



    def _timer_cb_nomad(self):
        
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)

        batch_obs_imgs = []
        batch_goal_data = []

        crop = True
        
        start_time = time.time()
        # Transform observation once
        transf_obs_img = transform_images(
            list(self.context_queue), self.model_params["image_size"], center_crop=crop
        )

        # Vectorized goal processing
        goal_imgs = self.topomap[start:end + 1]  
        batch_goal_data_np = np.concatenate([
            transform_images(sg_img, self.model_params["image_size"], center_crop=crop)
            for sg_img in goal_imgs
        ], axis=0).astype('float32')

        # Repeat observation for batch
        num_goals = len(goal_imgs)
        batch_obs_imgs_np = np.tile(transf_obs_img, (num_goals, 1, 1, 1)).astype('float32')
        input_goal_mask_np = np.zeros((num_goals,), dtype=np.int64)
               

        ort_inputs = {
            "obs_img": batch_obs_imgs_np.astype(np.float32),
            "goal_img": batch_goal_data_np.astype(np.float32),
            "input_goal_mask": input_goal_mask_np.astype(np.int64),
        }
        

        # To handle garbage collect and double free corruption C error
        try:
            obsgoal_cond = self.vis_encoder.run(None, ort_inputs)[0]
        except Exception as e:
            self.get_logger().error(f"Inference failed vis_encoder: {e}")
        except KeyboardInterrupt:
            self.get_logger().info("Inference vis_encoder interrupted by user.")

        ort_inputs = {
            "obsgoal_cond": obsgoal_cond,
        }
        
        try:
            distances =  self.dist_pred.run(None, ort_inputs)[0]
        except Exception as e:
            self.get_logger().error(f"Inference failed dist_pred: {e}")
        except KeyboardInterrupt:
            self.get_logger().info("Inference dist_pred interrupted by user.")

        min_dist_idx = np.argmin(distances)

        self.closest_node = min_dist_idx + start
        print("closest node:", self.closest_node)

        closest_node_msg = Int32()
        closest_node_msg.data = int(self.closest_node)
        self.closest_node_pub.publish(closest_node_msg)
        
        sg_idx = min(min_dist_idx + int(distances[min_dist_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
        obs_cond_np = obsgoal_cond[sg_idx]


        # infer action
        with torch.no_grad():
            # encoder vision features
            if len(obs_cond_np.shape) == 2:
                obs_cond_np = np.tile(obs_cond_np, (self.args.num_samples, 1))
            else:
                obs_cond_np = np.tile(obs_cond_np, (self.args.num_samples, 1, 1))

            # we need eq. global_cond torch.Size([8, 256])
            if obs_cond_np.ndim == 3 and obs_cond_np.shape[1] == 1:
                obs_cond_np = obs_cond_np.squeeze(1)

            
            # initialize action from Gaussian noise
            naction_np = np.random.randn(
                self.args.num_samples, self.model_params["len_traj_pred"], 2
            ).astype(np.float32)

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            start_time = time.time() 
            
            for k in self.noise_scheduler.timesteps[:]:
                # predict noise
                k_np = np.array(k.cpu().item(), dtype=np.int64)
                ort_sess_noise_pred_inputs = {
                    "sample": naction_np,   
                    "timestep": k_np,
                    "global_cond": obs_cond_np,
                }

                try:
                    noise_pred = self.noise_pred.run(None, ort_sess_noise_pred_inputs)[0]
                except Exception as e:
                    self.get_logger().error(f"Inference failed noise_pred: {e}")
                except KeyboardInterrupt:
                    self.get_logger().info("Inference noise_pred interrupted by user.")

                # DDPMScheduler need torch tensors (@TODO find a numpy implementation?)
                noise_pred_torch = torch.from_numpy(noise_pred).float().to(self.device)
                naction_torch = torch.from_numpy(naction_np).float().to(self.device)
                # print("before noise scheduler")
                naction_torch = self.noise_scheduler.step(
                    model_output=noise_pred_torch,
                    timestep=k,
                    sample=naction_torch
                ).prev_sample
                naction_np = naction_torch.detach().cpu().numpy()

            inference_time = time.time() - start_time
            self.get_logger().info(f"Inference time: {inference_time:.3f} seconds")

            inference_time_msg = Float32()
            inference_time_msg.data = inference_time
            self.inference_pub.publish(inference_time_msg)

        naction_np = to_numpy(get_action(naction_torch))

        naction_np = naction_np[0] 
        chosen_waypoint = naction_np[self.args.waypoint]


        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_pub.publish(waypoint_msg)

        self.get_logger().info(f"Closest node: {self.closest_node}")
        reached_goal = bool(self.closest_node == self.goal_node)
        self.goal_pub.publish(Bool(data=reached_goal))


    def _publish_goal_images(self, sg_img: PILImage.Image, goal_img: PILImage.Image):
        """Publish current sub‑goal and final goal images as ROS sensor_msgs/Image."""
        for img, pub in [(sg_img, self.subgoal_pub), (goal_img, self.goal_pub_img)]:
            cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)

    def _publish_viz_image(self, traj_batch: np.ndarray):
        frame = np.array(self.context_queue[-1])  # latest RGB frame
        img_h, img_w = frame.shape[:2]
        viz = frame.copy()

        cx = img_w // 2
        cy = int(img_h * 0.95)

        pixels_per_m = 3.0
        lateral_scale = 1.0
        robot_symbol_length = 10

        cv2.line(
            viz,
            (cx - robot_symbol_length, cy),
            (cx + robot_symbol_length, cy),
            (255, 0, 0),
            2,
        )
        cv2.line(
            viz,
            (cx, cy - robot_symbol_length),
            (cx, cy + robot_symbol_length),
            (255, 0, 0),
            2,
        )

        # Draw each trajectory
        for i, traj in enumerate(traj_batch):
            pts = []
            pts.append((cx, cy))

            acc_x, acc_y = 0.0, 0.0
            for dx, dy in traj:
                acc_x += dx
                acc_y += dy
                px = int(cx - acc_y * pixels_per_m * lateral_scale)
                py = int(cy - acc_x * pixels_per_m)
                pts.append((px, py))

            if len(pts) >= 2:
                color = (
                    (0, 255, 0) if i == 0 else (255, 200, 0)
                )
                cv2.polylines(viz, [np.array(pts, dtype=np.int32)], False, color, 2)

        img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(img_msg)


def main():
    parser = argparse.ArgumentParser("Topological navigation (ROS 2)")
    parser.add_argument(
        "--dir", "-d", default="mist_office_new_chair", help="sub‑directory under ../topomaps/images/"
    )
    parser.add_argument(
        "--goal-node", "-g", type=int, default=-1, help="Goal node index (-1 = last)"
    )
    
    parser.add_argument("--waypoint", "-w", type=int, default=2)
    parser.add_argument("--close-threshold", "-t", type=float, default=3)
    parser.add_argument("--radius", "-r", type=int, default=4)
    parser.add_argument("--num-samples", "-n", type=int, default=8)

    args = parser.parse_args()

    rclpy.init()
    node = None
    try:
        node = NavigationNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Shutdown] KeyboardInterrupt received...")
    except Exception as e:
        print(f"\n[Shutdown] Unexpected error occurred: {e}")
        if node is not None:
            # 1. Stop all timers/subscriptions/publishers first
            node.destroy_node()
            node.model = None  # Explicitly release model resources before shutdown
            # 2. Delete the node object entirely to drop references
            del node
            
        # 3. Clear model from memory before rclpy.shutdown
        gc.collect()
        
        if rclpy.ok():
            rclpy.shutdown()

        os._exit(0)


if __name__ == "__main__":
    main()