import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

import onnxruntime as ort
import onnx

# ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Bool, Float32MultiArray, Int32
from nav_msgs.msg import Path
from utils import msg_to_pil, to_numpy, transform_images, load_model

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
                        SAMPLED_ACTIONS_TOPIC,
                        CLOSEST_NODE_TOPIC)


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 
VEL_TOPIC = robot_config["vel_navi_topic"]

# GLOBALS
context_queue = []
context_size = None  
subgoal = []

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
    assert context_size != None

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

    
     # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    waypoint_viz_pub = rospy.Publisher(
        "viz_wp", PoseStamped, queue_size=1)
    path_viz_pub = rospy.Publisher(
        "viz_path", Path, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    goal_img_pub = rospy.Publisher("/topoplan/goal_img", Image, queue_size=1)
    subgoal_img_pub = rospy.Publisher("/topoplan/subgoal_img", Image, queue_size=1)
    closest_node_img_pub = rospy.Publisher("/topoplan/closest_node_img", Image, queue_size=1)
    closest_node_pub = rospy.Publisher(CLOSEST_NODE_TOPIC, Int32, queue_size=10)

    # Try onnx vint
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession('../model_weights/dist_pred_net.onnx', providers=providers)

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    # navigation loop
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                closest_node_msg = Int32()
                closest_node_msg.data = closest_node
                closest_node_pub.publish(closest_node_msg)
                
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
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
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] 
                chosen_waypoint = naction[args.waypoint]
            else: # THIS IS NOT NOAMD SO VINT OR GNM ? Its seems its using subgoal (Vint paper talked about subgoal -> subgoal candidates)
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                batch_obs_imgs_np = []
                batch_goal_data_np = []
                
                crop=True
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"], center_crop=crop)
                    goal_data = transform_images(sg_img, model_params["image_size"], center_crop=crop)
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)

                    batch_obs_imgs_np.append(to_numpy(transf_obs_img))
                    batch_goal_data_np.append(to_numpy(goal_data))
                    
                # predict distances and waypoints
                batch_obs_imgs_gpu = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data_gpu = torch.cat(batch_goal_data, dim=0).to(device)
                print("batch_obs_imgs shape:", batch_obs_imgs_gpu.shape)
                print("batch_goal_data shape:", batch_goal_data_gpu.shape)

                time_0 = time.time()
                distances, waypoints = model(batch_obs_imgs_gpu, batch_goal_data_gpu)
                print(f'Inference time with torch {time.time() - time_0}')
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                
                batch_obs_imgs_np = np.concatenate(batch_obs_imgs_np, axis=0)
                batch_goal_data_np = np.concatenate(batch_goal_data_np, axis=0)
                print("batch_obs_imgs shape:", batch_obs_imgs_np.shape)
                print("batch_goal_data shape:", batch_goal_data_np.shape)
                ort_inputs = {
                    "obs": batch_obs_imgs_np,
                    "goal": batch_goal_data_np,
                }
                time_0 = time.time()
                ort_outputs = ort_session.run(None, ort_inputs)
                print(f'Inference time without torch {time.time() - time_0}')
                print("Available providers:", ort.get_available_providers())
                print("Session providers:", ort_session.get_providers())


                max_diff_model = abs(distances - ort_outputs[0]).max()
                print(
                    f"Maximum difference between PyTorch and ONNX: {max_diff_model}"
                )

                print("distances shape:", distances.shape, "len:", distances)
                print("waypoints shape:", waypoints.shape, "len:", waypoints)

                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                print("min dist idx:", min_dist_idx, "min dist:", distances[min_dist_idx], "close_threshold:", args.close_threshold)
                if distances[min_dist_idx] > args.close_threshold:
                    print("Not close enough to the next node, choosing closest waypoint", waypoints[min_dist_idx][args.waypoint], "at index", min_dist_idx)
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    print("Very far already a lost cause ", min(min_dist_idx + 1, len(waypoints) - 1))
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                    print("closest start", start, "min_dist_idx + 1", min_dist_idx + 1, "goal_node", goal_node)
                    closest_node = min(start + min_dist_idx + 1, goal_node)
                # print("chosen wp", chosen_waypoint)
                print("min dist idx", min_dist_idx)

                print("closest node", closest_node)

                print(f"end {end} start {start}")
                # Publish visualization messages
                # Waypoint
                waypoint_msg_viz = PoseStamped()
                waypoint_msg_viz.header.frame_id = "odom"
                waypoint_msg_viz.header.stamp = rospy.Time.now()
                wp_point = Point(x=chosen_waypoint[0], y=chosen_waypoint[1])
                # print("waypoint point:", wp_point)
                wp_position = Pose(position=wp_point)
                # print("waypoint position:", wp_position)
                waypoint_msg_viz.pose = wp_position           
                waypoint_viz_pub.publish(waypoint_msg_viz)

                # Path
                path_msg_viz = Path()
                path_msg_viz.header.frame_id = "base_footprint"
                path_msg_viz.header.stamp = rospy.Time.now()
                print("------")
                for wp in waypoints[min_dist_idx]:
                    # print("waypoint:", wp)
                    path_msg_viz.poses.append(PoseStamped(
                        pose=Pose(position=Point(x=wp[0], y=wp[1]))))
                path_viz_pub.publish(path_msg_viz)
                # for dist in distances:
                    # print("distance:", dist)

        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)

        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
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
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=0.5,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=2,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
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







