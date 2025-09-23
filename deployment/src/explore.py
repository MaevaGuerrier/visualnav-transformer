
import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
from utils import msg_to_pil, to_numpy, transform_images, load_model
from viz_utils import viz_chosen_wp, make_marker_array


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
# TODO THIS HAS TO BE IN CONFIG FILE
VIZ_IMAGE_SIZE_FISHEYE = (640, 480) # (640, 480) orig fisheye image size

# GLOBALS
context_queue = []
context_size = None  
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

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")

def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


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



def main(args: argparse.Namespace):
    global context_size

    model, model_params = _load_model(args.model, args.device)
    context_size = model_params["context_size"]

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    # Publisher for pd_controller
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    
    # RVIZ diffusion paths and choosen waypoint
    chosen_wp_viz_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    all_path_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
    

    # Images overlay
    cam_wp_pub = rospy.Publisher("/wps_overlay_img", Image, queue_size=10) # not corrected action
    # cam_corr_wp_pub = rospy.Publisher("/topoplan/wps_corrected_overlay_img", Image, queue_size=10)
    trav_wp_pub = rospy.Publisher("/wps_overlay_trav_img", Image, queue_size=10) # not corrected action
    # trav_corr_wp_pub = rospy.Publisher("/topoplan/wps_corrected_overlay_trav_img", Image, queue_size=10)




    rospy.loginfo("Registered with master node. Waiting for image observations...")

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
                rospy.loginfo(f"time elapsed: {time.time() - start_time}")

            naction = to_numpy(get_action(naction))


            rospy.logdebug(f"naction {naction}")

            naction_selected = naction[0] # change this based on heuristic 

            rospy.logdebug(f"naction[0] {naction[0]}")

            chosen_waypoint = naction_selected[args.waypoint]
            rospy.loginfo(f"chosen waypoint {chosen_waypoint}")
            viz_chosen_wp(chosen_waypoint, chosen_wp_viz_pub)

            if model_params["normalize"]:
                chosen_waypoint *= (MAX_V / RATE)
            waypoint_msg.data = chosen_waypoint
            waypoint_pub.publish(waypoint_msg)
            rospy.loginfo(f"Waypoint published to pd_controller using topic: {WAYPOINT_TOPIC}")
        
        
            make_marker_array(naction, all_path_pub)
        
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


