#TODO DONE Listen to the topic "/wild_visual_navigation_node/{cam}/traversability" 
#TODO DONE Understand image values and how they relate to traversability
# See if make sense to use the torch map function to get a image gradient 
## understand the image gradient 
# Evaluate the cost of one pixel in the image gradient
# Evaluate the cost of a line in the image gradient
# See how to inflate to take into account the robot size
# Replicate the diffusion head correction




# TODO I CREATE A UTILS VIZ I HAVE TO REFACTOR 


#!/usr/bin/env python3
import yaml
import argparse
import os
import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Any, Deque
from collections import deque
import time

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

# torch
import torch
import torch.nn.functional as F

# Utils
from topic_names import (IMAGE_TOPIC, WAYPOINT_TOPIC, REACHED_GOAL_TOPIC)                 
from utils import load_model, msg_to_pil, transform_images, to_numpy, pil_to_numpy_array
from viz_utils import publish_overlay_image, viz_chosen_wp, make_marker_array
from trav_utils import select_traj_best_traversability, traversabilityImageSubscriber


# Diffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.training.train_utils import get_action

# Constants variables
## Paths
TOPOMAP_IMAGES_DIR = "../topomaps/images"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
## Robot parameters
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

VIZ_IMAGE_SIZE_FISHEYE = (640, 480) # (640, 480) orig fisheye image size


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

def main(args: argparse.Namespace):
    rospy.init_node("traversability_diffusor", anonymous=True, log_level=args.log_level)
    trav_img_subscriber = traversabilityImageSubscriber()


    # PUBLISHERS
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1) 
    goal_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1)
    chosen_wp_viz_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    all_path_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
    # OVERLAY IMAGE
    # TODO BETTER NAMING TO UNDERSTAND
    cam_wp_pub = rospy.Publisher("/wps_overlay_img", Image, queue_size=10) # not corrected action
    # cam_corr_wp_pub = rospy.Publisher("/topoplan/wps_corrected_overlay_img", Image, queue_size=10)
    trav_wp_pub = rospy.Publisher("/wps_overlay_trav_img", Image, queue_size=10) # not corrected action
    # trav_corr_wp_pub = rospy.Publisher("/topoplan/wps_corrected_overlay_trav_img", Image, queue_size=10)

    rate = rospy.Rate(RATE)

    topomap, goal_node = _load_topomap(args.dir, args.goal_node)
    closest_node = 0
    model, model_params = _load_model(args.model, args.device)
    context_size = model_params["context_size"]
    rospy.logdebug(f"Model context size: {context_size}")
    assert context_size != None
    context_queue: Deque[np.ndarray] = deque(maxlen=context_size + 1)

    # TODO CLEANUP THIS MAKE A CLASS ONCE ALL IS GOOD
    def _callback_obs(msg):
        rospy.logdebug("Received observation image")
        context_queue.append(msg_to_pil(msg))
        rospy.logdebug(f"Context queue size: {len(context_queue)}")

    rospy.Subscriber( IMAGE_TOPIC, Image, _callback_obs, queue_size=1)

    
    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    while not rospy.is_shutdown():

        if len(context_queue) > model_params["context_size"]:

            obs_images = transform_images(list(context_queue), model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1) 
            obs_images = obs_images.to(args.device)
            mask = torch.zeros(1).long().to(args.device)

            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(args.device) for g_img in topomap[start:end + 1]]
            goal_image = torch.concat(goal_image, dim=0)

            obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
            dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            min_idx = np.argmin(dists)
            closest_node = min_idx + start
            rospy.logdebug(f"Closest node: {closest_node} Goal node: {goal_node}")
            sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
            obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

            with torch.no_grad():
                # encoder vision features
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                
                # initialize action from Gaussian noise
                naction = torch.randn(
                    (args.num_samples, model_params["len_traj_pred"], 2), device=args.device)
                
                noise_scheduler.set_timesteps(num_diffusion_iters)

                start_time = time.time()
                for timestep in noise_scheduler.timesteps[:]:
                        
                    # predict noise
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=timestep,
                        global_cond=obs_cond
                    )
                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=timestep,
                        sample=naction
                    ).prev_sample

                rospy.logdebug(f"time elapsed: {time.time() - start_time}")

            naction = to_numpy(get_action(naction))
            naction_selected = naction[0] # we could choose based on trav instead 

            if args.trav_baseline:
                best_traj, best_score = select_traj_best_traversability(
                    trav_img=trav_img_subscriber.get_trav_img(),
                    camera_matrix=camera_matrix_orig,
                    dist_coeffs=dist_coeffs,
                    list_trajs=naction,
                    viz_img_size=VIZ_IMAGE_SIZE_FISHEYE,
                    resize_factor=True,
                )
                rospy.logdebug(f"Best trajectory score: {best_score}")

                if best_traj is not None:
                    naction_selected = best_traj

            chosen_waypoint = naction_selected[args.waypoint] 
            rospy.logdebug(f"Chosen waypoint: {chosen_waypoint}")             


            if model_params["normalize"]:
                chosen_waypoint[:2] *= (MAX_V / RATE)  
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint
            waypoint_pub.publish(waypoint_msg)
            rospy.logdebug(f"Published waypoint: {chosen_waypoint}")
            viz_chosen_wp(chosen_waypoint, chosen_wp_viz_pub)

            img = context_queue[-1]
            img = pil_to_numpy_array(image_input=img, target_size=VIZ_IMAGE_SIZE_FISHEYE)
            publish_overlay_image(camera_matrix_orig, dist_coeffs, img, cam_wp_pub, naction, viz_img_size=VIZ_IMAGE_SIZE_FISHEYE)

            overlay_traj_img = trav_img_subscriber.get_overlay_traj_img()
            if overlay_traj_img is not None:
                rospy.logdebug(f"Publishing traversability overlay image with trajectories using overlay_traj of shape {overlay_traj_img.shape}")
                publish_overlay_image(camera_matrix_orig, dist_coeffs, overlay_traj_img, trav_wp_pub, naction, viz_img_size=VIZ_IMAGE_SIZE_FISHEYE, resize_factor=True) # ORIG ACTION WITHOUT CORRECTION


            make_marker_array(naction, all_path_pub)

            reached_goal = closest_node == goal_node
            goal_pub.publish(reached_goal)
            if reached_goal:
                rospy.loginfo("Reached goal! Stopping...")

        rate.sleep()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    argparser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    argparser.add_argument(
        "--dir",
        "-d",
        default="new_lab",
        type=str,
        help="path to topomap images",
    )
    argparser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    argparser.add_argument(
        "--close-threshold",
        "-t",
        default=0.5,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    argparser.add_argument(
        "--radius",
        "-r",
        default=2,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    argparser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )

    argparser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose logging"
    )

    argparser.add_argument(
        "--trav_baseline", action="store_true", help="Enable traversability baseline ONLY"
    )
    
    args = argparser.parse_args()
    if args.debug:
        args.log_level = rospy.DEBUG
    else:
        args.log_level = rospy.INFO


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.loginfo(
        f"Log level set to: {args.log_level}\n"
        f"Using device: {args.device}\n"
        f"_____________________________________________\n"
        f"Listening to image topic {IMAGE_TOPIC} \n Publishing to topic {robot_config['vel_navi_topic']} with observation rate at {robot_config['frame_rate']} Hz"
    )

    try:
        main(args)
    except rospy.ROSInterruptException:
        pass
