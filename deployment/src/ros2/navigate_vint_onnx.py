from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, List
import onnxruntime as ort

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


# UTILS
from src.topic_names import (
    IMAGE_TOPIC,
    WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    CLOSEST_NODE_TOPIC,
)

# CONSTANTS
WORK_DIR = "/workspace/src/visualnav-transformer/deployment/" # ALWAYS DEPLOY INSIDE DOCKER
TOPOMAP_IMAGES_DIR = f"{WORK_DIR}src/topomaps/images" #/home/indro/SafeGNM/src/visualnav-transformer/deployment/src/topomaps/images
MODEL_WEIGHTS_PATH = f"{WORK_DIR}model_weights/"
ROBOT_CONFIG_PATH =f"{WORK_DIR}config/robot.yaml"
MODEL_CONFIG_PATH = f"{WORK_DIR}../train/config/"

with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE = ROBOT_CONF["frame_rate"]  # Hz

def _load_model():
    model_params = {"normalize": True, "context_size": 5, "image_size": [85, 64]}
    model = load_model_onnx("vint")

    return model, model_params


class NavigationNode(Node):
    """Sub‑goal navigation with topomap + trajectory visualisation."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("navigation")
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.model, self.model_params = _load_model()

        self.context_size: int = self.model_params["context_size"]

        self.bridge = CvBridge()
        self.context_queue = deque(maxlen=self.context_size + 1)
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
        self.topomap, self.goal_node = self._load_topomap(args.dir, args.goal_node)
        self.closest_node = 0

        self.create_subscription(Image, IMAGE_TOPIC, self._image_cb, 1)
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
        self.get_logger().info("DEPTH MODEL CONFIGURATION:")
        self.get_logger().info(f"  - UniDepth model: UniDepthV2")
        self.get_logger().info(
            f"  - Pretrained weights: lpiccinelli/unidepth-v2-vits14"
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

    def _timer_cb(self):
        if len(self.context_queue) <= self.context_size:
            return

        self._timer_cb_other()

        if self.closest_node == self.goal_node:
            self.get_logger().info("Reached goal! Stopping...")

    def _timer_cb_other(self):
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
        

        distances, waypoints = self.model.run(None, {
            "obs_img": batch_obs_imgs_np,
            "goal_img": batch_goal_data_np,
        })
        inference_time = time.time() - start_time
        self.get_logger().info(f"Inference time: {inference_time:.3f} seconds")

        inference_time_msg = Float32()
        inference_time_msg.data = inference_time
        self.inference_pub.publish(inference_time_msg)


        # look for closest node
        min_dist_idx = np.argmin(distances)

        if distances[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
            self.closest_node = start + min_dist_idx
        else:
            chosen_waypoint = waypoints[
                min(min_dist_idx + 1, len(waypoints) - 1)
            ][self.args.waypoint]

            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)

        closest_node_msg = Int32()
        closest_node_msg.data = int(self.closest_node)
        self.closest_node_pub.publish(closest_node_msg)

        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_pub.publish(waypoint_msg)

        self.get_logger().info(f"Closest node: {self.closest_node}")
        reached_goal = bool(self.closest_node == self.goal_node)
        self.goal_pub.publish(Bool(data=reached_goal))

        sg_global_idx = min(
            start
            + min_dist_idx
            + int(distances[min_dist_idx] <= self.args.close_threshold),
            self.goal_node,
        )
        sg_pil = self.topomap[sg_global_idx]
        goal_pil = self.topomap[self.goal_node]

        self._publish_goal_images(sg_pil, goal_pil)

    def _publish_goal_images(self, sg_img: PILImage.Image, goal_img: PILImage.Image):
        """Publish current sub‑goal and final goal images as ROS sensor_msgs/Image."""
        for img, pub in [(sg_img, self.subgoal_pub), (goal_img, self.goal_pub_img)]:
            cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)


    def _publish_msgs(self, traj_batch: np.ndarray):
        # sampled actions
        actions_msg = Float32MultiArray()
        actions_msg.data = [0.0] + [float(x) for x in traj_batch.flatten()]
        self.sampled_actions_pub.publish(actions_msg)

        # chosen waypoint
        chosen = traj_batch[0][self.args.waypoint]
        if self.model_params.get("normalize", False):
            chosen *= MAX_V / RATE
        wp_msg = Float32MultiArray()
        wp_msg.data = [float(chosen[0]), float(chosen[1]), 0.0, 0.0]  # 4‑D compat
        self.waypoint_pub.publish(wp_msg)

        # goal status
        self.get_logger().info(f"Closest node: {self.closest_node}")
        
        reached = bool(self.closest_node == self.goal_node)
        self.goal_pub.publish(Bool(data=reached))

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
    parser.add_argument("--close-threshold", "-t", type=float, default=0.5)
    parser.add_argument("--radius", "-r", type=int, default=2)
    parser.add_argument("--num-samples", "-n", type=int, default=8)

    args = parser.parse_args()

    rclpy.init()
    node = NavigationNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()