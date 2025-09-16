#TODO DONE Listen to the topic "/wild_visual_navigation_node/{cam}/traversability" 
#TODO DONE Understand image values and how they relate to traversability
# See if make sense to use the torch map function to get a image gradient 
## understand the image gradient 
# Evaluate the cost of one pixel in the image gradient
# Evaluate the cost of a line in the image gradient
# See how to inflate to take into account the robot size
# Replicate the diffusion head correction

#!/usr/bin/env python3
import yaml
import argparse
import os
import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Any
# ROS
import rospy
from sensor_msgs.msg import Image
import ros_numpy
# torch
import torch
# Utils
from topic_names import (IMAGE_TOPIC)                 
from utils import load_model, msg_to_pil, transform_images, to_numpy
# Diffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


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


# Global variables (see if there is a better way to do this)
context_queue = []
context_size = None
trav_img = None


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


def _get_trajs_traversability():
    if trav_img is None:
        return 
    



# Callbacks


# 0 is untraversable and 1 is fully traversable. https://arxiv.org/pdf/2404.07110
def _callback_traversability_image(trav_img_msg: Image):
    trav_img = ros_numpy.numpify(trav_img_msg) # like cv_bridge but without requiring cv2 dep
    # TODO CHECK 1 Traversability images values need to remains unchanged here
    rospy.logdebug(f"Traversability value debug (array from ros img) {trav_img[0]}")
    trav_img = torch.from_numpy(trav_img)
    # TODO CHECK 2 Traversability images values need to remains unchanged here
    rospy.logdebug(f"Received traversability image of shape: {trav_img.shape}")
    rospy.logdebug(f"traversability image data type: {trav_img.dtype}")

def _callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


def main(args: argparse.Namespace):
    rospy.init_node("traversability_diffusor", anonymous=True, log_level=args.log_level)

    rospy.Subscriber("/wild_visual_navigation_node/front/traversability", Image, _callback_traversability_image) 
    rospy.Subscriber( IMAGE_TOPIC, Image, _callback_obs, queue_size=1)

    rate = rospy.Rate(RATE)

    topomap, goal_node = _load_topomap(args.dir, args.goal_node)
    closest_node = 0
    model, model_params = _load_model(args.model, args.device)
    
    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    while not rospy.is_shutdown():

        
        if len(context_queue) > model_params["context_size"]:

            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1) 
            obs_images = obs_images.to(args.device)

            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(args.device) for g_img in topomap[start:end + 1]]
            goal_image = torch.concat(goal_image, dim=0)

            obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
            dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            min_idx = np.argmin(dists)
            closest_node = min_idx + start
            rospy.logdebug("Closest node:", closest_node)
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
                
                    interval = 6
                    if timestep <= interval:
                        grad = pathguide.get_gradient(naction, goal_pos=rela_pos, scale_factor=scale_factor)
                        naction -= grad


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
        default="topomap",
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
