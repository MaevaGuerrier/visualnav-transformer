
import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

from PIL import Image as PILImage
from typing import List, Tuple, Dict, Any, Deque
from collections import deque
import cv2
from cv_bridge import CvBridge
import time

# ROS
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model
# import ros_numpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time


# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)


# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
context_queue = []
context_size = None  

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

VIZ_IMAGE_SIZE_FISHEYE = (640, 480) # (640, 480) orig fisheye image size
VIZ_IMAGE_SIZE_TRAV = (224, 224)



# TODO 
camera_matrix_orig = np.array([
    [262.459286,   1.916160, 327.699961],
    [  0.000000, 263.419908, 224.459372],
    [  0.000000,   0.000000,   1.000000]
], dtype=np.float64)

dist_coeffs = np.array([
    -0.03727222045233312, 
        0.007588870705292973,
    -0.01666117486022043, 
        0.00581938967971292
], dtype=np.float64)

# Global variables (see if there is a better way to do this)

trav_img = None
bridge = CvBridge()


def _load_model(model_name: str, device: torch.device, train: bool = False)-> Tuple["Model", Dict[str, Any]]:
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_conf_path = model_paths[model_name]["config_path"]

    ckpt_path = model_paths[model_name]["ckpt_path"]
    with open(model_conf_path, "r") as f:
        model_params = yaml.safe_load(f)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

    rospy.loginfo(f"Loading model navigation from {ckpt_path}")
    model = load_model(ckpt_path, model_params, device).to(device)
    if train:
        model.train()
        rospy.logwarn("Model set to train mode!")
    else:
        model.eval()
    return model, model_params

def _load_topomap(dir_path: str, goal_node: int) -> Tuple[List[PILImage.Image], int]:
    topomap_filenames = sorted(os.listdir(os.path.join(
    TOPOMAP_IMAGES_DIR, dir_path)), key=lambda x: int(x.split(".")[0]))
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



# VISUALIZATION


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = np.zeros((3, 1), dtype=np.float64)

    xyz[..., 0] += camera_x_offset

    # Convert from (x, y, z) to (y, -z, x) for cv2
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    
    # done for cv2.fisheye.projectPoint requires float32/float64 and shape (N,1,3),
    xyz_cv = xyz_cv.reshape(batch_size * horizon, 1, 3).astype(np.float64)


    # uv, _ = cv2.projectPoints(
    #     xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    # )
    uv, _ = cv2.fisheye.projectPoints(
        xyz_cv, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    uv = uv.reshape(batch_size, horizon, 2)
    
    
    return uv

def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    viz_img_size: Tuple[int, int],
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    # print(pixels)
    # Flip image horizontally
    pixels[:, 0] = viz_img_size[0] - pixels[:, 0]

    return pixels

def plot_trajs_and_points_on_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
    viz_img_size: Tuple[int, int],
    resize_factor:bool=False, 
):
    """
    Plot trajectories and points on an image.
    resize_factor: if True resize the image to viz_img_size. This is needed due to the fact that orginal image coming from fisheye is 640 x 480 and the traversability image is 224 x 224.
    Thus the camera matrix needs to be scaled accordingly.
    """
    camera_height = 0.25
    camera_x_offset = 0.10

    # if resize_factor:
    #     camera_matrix[0,0] *= .35
    #     camera_matrix[0,2] *= .35
    #     camera_matrix[1,1] *= .46
    #     camera_matrix[1,2] *= .46

    for i, traj in enumerate(list_trajs):
        xy_coords = traj[:, :2]
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, viz_img_size
        )
        
        if resize_factor:
            traj_pixels[:,0] *= .35
            traj_pixels[:,1] *= .46

        # print(traj_pixels)
                
        valid = (
            (traj_pixels[:, 0] >= 0) & (traj_pixels[:, 0] < viz_img_size[1]) &
            (traj_pixels[:, 1] >= 0) & (traj_pixels[:, 1] < viz_img_size[0])
        )

        # valid is a boolean mask for each pixel
        inside_pixels = traj_pixels[valid]

        # Check if ALL are inside
        all_inside = np.all(valid)

        # Check if ANY are inside
        any_inside = np.any(valid)

        # print(f"Trajectory {i}: all inside: {all_inside}, any inside: {any_inside}, total points: {len(traj_pixels)}, inside points: {len(inside_pixels)}")
        
        points = traj_pixels.astype(int).reshape(-1, 1, 2)
        # print(f"points shape {points.shape}, traj_pixels shape {traj_pixels.shape}")
        # print(points[0])
        # print(points[:, :, ::-1][0])
        # points = points[:, :, ::-1]
        # Random color for each trajectory
        color = tuple(int(x) for x in np.random.choice(range(50, 255), size=3))

        # inverting x,y axis so origin in image is down-left corner
        if resize_factor:
            points[:, :, 1] = viz_img_size[1] * .46  - 1 - points[:, :, 1]
        else:
            points[:, :, 1] = viz_img_size[1] - 1 - points[:, :, 1]

        # Draw trajectory
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=3)

        # Draw start point (green) and goal point (red)
        # start = tuple(points[0, 0])
        # goal = tuple(points[-1, 0])
        # cv2.circle(img, start, 6, (0, 255, 0), -1)
        # cv2.circle(img, goal, 6, (0, 0, 255), -1)

    return img

def make_path_marker(points, marker_id, r, g, b, frame_id="base_link"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "multi_paths"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.scale.x = 0.05  # line width
    marker.color.a = 1.0
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b

    # print("---------------")
    for (x, y) in points:
        p = Point()
        # print(f"x {x} y {y}")
        p.x, p.y, p.z = x, y, 0.0
        marker.points.append(p)
    # print("---------------")
    return marker

def viz_chosen_wp(chosen_waypoint, waypoint_viz_pub):
    marker = Marker()
    marker.header.frame_id = "base_link"   # or "odom", "base_link" depending on your TF
    marker.header.stamp = rospy.Time.now()

    marker.ns = "points"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    # Example 2D point (x, y, z=0)
    marker.pose.position.x = chosen_waypoint[0]
    marker.pose.position.y = chosen_waypoint[1]
    marker.pose.position.z = 0.0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Sphere size
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    # Color (red)
    marker.color.a = 1.0  # alpha
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    waypoint_viz_pub.publish(marker)





def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


def main(args: argparse.Namespace):
    global context_size

    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    path_viz_pub = rospy.Publisher(
        "viz_path", Path, queue_size=1)
    chosen_wp_viz_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    all_path_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)
    


    print("Registered with master node. Waiting for image observations...")

    while not rospy.is_shutdown():
        # EXPLORATION MODE
        waypoint_msg = Float32MultiArray()
        if (
                len(context_queue) > model_params["context_size"]
            ):

            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=True)
            obs_images = obs_images.to(device)
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
            mask = torch.ones(1).long().to(device) # ignore the goal

            # infer action
            with torch.no_grad():
                # encoder vision features
                obs_cond = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
                
                # (B, obs_horizon * obs_dim)
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                
                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                start_time = time.time()
                for k in noise_scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                print("time elapsed:", time.time() - start_time)

            naction = to_numpy(get_action(naction))
            
            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
            sampled_actions_pub.publish(sampled_actions_msg)

            print(f"naction {naction}")

            naction = naction[0] # change this based on heuristic

            print(f"naction[0] {naction[0]}")

            chosen_waypoint = naction[args.waypoint]
            print(f"chosen waypoint {chosen_waypoint}")
            # viz_chosen_wp(chosen_waypoint, chosen_wp_viz_pub)

            # ma = MarkerArray()
            # for idx, paths in enumerate(naction):
            #     r = 0.0
            #     g = 0.0
            #     b = 1.0
            #     marker = make_path_marker(
            #         paths, idx, r, g, b, frame_id="base_link")
            #     ma.markers.append(marker)
            # all_path_pub.publish(ma) 

            # Path
            path_msg_viz = Path()
            path_msg_viz.header.frame_id = "base_link"
            path_msg_viz.header.stamp = rospy.Time.now()
            path_msg_viz.poses.append(PoseStamped(pose=Pose(position=Point(x=chosen_waypoint[0], y=chosen_waypoint[1]))))
            path_viz_pub.publish(path_msg_viz)

            if model_params["normalize"]:
                chosen_waypoint *= (MAX_V / RATE)
            waypoint_msg.data = chosen_waypoint
            waypoint_pub.publish(waypoint_msg)
            print("Published waypoint")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)


