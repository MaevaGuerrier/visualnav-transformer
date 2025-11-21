import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import onnxruntime as ort

# ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Bool, Float32MultiArray, Int32
from nav_msgs.msg import Path
from utils_onnx import msg_to_pil, transform_images, load_model_trt, load_model_onnx


# To DELETE AS WE CHECK THAT EACH TRT MODULE WORKS CORRECTLY -----------------------------
from utils import load_model, to_numpy
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# --------------------------------------------------------------------------------

# from vint_train.training.train_utils import get_action
# import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
from topic_names import (
    IMAGE_TOPIC,
    WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    CLOSEST_NODE_TOPIC,
)


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
VEL_TOPIC = robot_config["vel_navi_topic"]

model_params = {"normalize": True, "context_size": 5, "image_size": [85, 64]}

# GLOBALS
context_queue = []
context_size = model_params["context_size"]
subgoal = []

# Load the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)



# TO TAKE OUT AS WE CHECK THAT EACH TRT MODULE WORKS CORRECTLY ------------------------------------------------

import torch

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


# -------------------------------------------------------------------------------------------------------------


def main(args: argparse.Namespace):
    global context_size

    # TO TAKE OUT AS WE CHECK THAT EACH TRT MODULE WORKS CORRECTLY ------------------------------------------------

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths["nomad"]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    
    context_size = model_params["context_size"]
    assert context_size != None

    # load model weights
    ckpth_path = model_paths["nomad"]["ckpt_path"]
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

    # -------------------------------------------------------------------------------------------------------------

    # trt_vision_encoder = load_model_trt("nomad_vision_encoder.trt")
    ort_sess_vis_encoder = load_model_onnx("nomad_vision_encoder")
    print("loaded vision encoder onnx model")
    ort_sess_dist_pred = load_model_onnx("nomad_dist_pred_net")
    print("loaded distance predictor onnx model")
    # load topomap
    topomap_filenames = sorted(
        os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, args.dir)),
        key=lambda x: int(x.split(".")[0]),
    )
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
    image_curr_msg = rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    waypoint_viz_pub = rospy.Publisher("viz_wp", PoseStamped, queue_size=1)
    path_viz_pub = rospy.Publisher("viz_path", Path, queue_size=1)
    sampled_actions_pub = rospy.Publisher(
        SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1
    )
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    goal_img_pub = rospy.Publisher("/topoplan/goal_img", Image, queue_size=1)
    subgoal_img_pub = rospy.Publisher("/topoplan/subgoal_img", Image, queue_size=1)
    closest_node_img_pub = rospy.Publisher(
        "/topoplan/closest_node_img", Image, queue_size=1
    )
    closest_node_pub = rospy.Publisher(CLOSEST_NODE_TOPIC, Int32, queue_size=10)

    # navigation loop
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        # print("in ros")
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
                # print("init")
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                # batch_obs_imgs_np = []
                # batch_goal_data_np = []

                crop = False
                
                time_0 = time.time()
                # Transform observation once
                transf_obs_img = transform_images(
                    context_queue, model_params["image_size"], center_crop=crop
                )

                # Vectorized goal processing
                goal_imgs = topomap[start:end + 1]  
                batch_goal_data_np = np.concatenate([
                    transform_images(sg_img, model_params["image_size"], center_crop=crop)
                    for sg_img in goal_imgs
                ], axis=0).astype('float16')

                # Repeat observation for batch
                num_goals = len(goal_imgs)
                batch_obs_imgs_np = np.tile(transf_obs_img, (num_goals, 1, 1, 1)).astype('float16')
                input_goal_mask_np = np.zeros((num_goals,), dtype=np.int64)
                # print(f"type batch_obs_imgs_np {batch_obs_imgs_np.dtype}, batch_goal_data_np {batch_goal_data_np.dtype}")
                # print(f"len batch_obs_imgs_np {len(batch_obs_imgs_np)}, len batch_goal_data_np {len(batch_goal_data_np)}")

                # print("batch_obs_imgs shape:", batch_obs_imgs)
                # print("batch_goal_data shape:", batch_goal_data)
                # import pdb; pdb.set_trace()
                # ort_inputs = {
                #     "obs_img": batch_obs_imgs_np,
                #     "goal_img": batch_goal_data_np,
                # }

                # ort_outputs = ort_session.run(None, ort_inputs)
                # print(f"Inference time without torch {time.time() - time_0}")
                
                # distances, waypoints = ort_outputs[0], ort_outputs[1]
                # goal_image shape: torch.Size([4, 3, 96, 96]), 
                # obs image shape: torch.Size([4, 12, 96, 96]), 
                # mask shape: torch.Size([4])
                
                # obsgoal_cond = trt_vision_encoder.infer(obs_img=batch_obs_imgs_np, goal_img=batch_goal_data_np, input_goal_mask=input_goal_mask_np)
                
                ort_inputs = {
                    "obs_img": batch_obs_imgs_np.astype(np.float32),
                    "goal_img": batch_goal_data_np.astype(np.float32),
                    "input_goal_mask": input_goal_mask_np.astype(np.int64),
                }
                obsgoal_cond = ort_sess_vis_encoder.run(None, ort_inputs)[0]
                
                print(f"Vision encoder Inference time without torch {time.time() - time_0}")
                # print(obsgoal_cond)
                
                # distances = model("dist_pred_net", obsgoal_cond=torch_obsgoal_cond)
                ort_inputs = {
                    "obsgoal_cond": obsgoal_cond,
                }
                time_1 = time.time()
                distances =  ort_sess_dist_pred.run(None, ort_inputs)[0]
                print(f"Distance prediction Inference time without torch {time.time() - time_1}")
                # print("distances:", distances, distances.shape)
                min_dist_idx = np.argmin(distances)
                
               
# -----------------

                closest_node = min_dist_idx + start
                print("closest node:", closest_node)
                closest_node_msg = Int32()
                closest_node_msg.data = closest_node
                closest_node_pub.publish(closest_node_msg)
                
                sg_idx = min(min_dist_idx + int(distances[min_dist_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx]


                torch_obs_cond = torch.from_numpy(np.asarray(obs_cond)).to(device)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = torch_obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = torch_obs_cond.repeat(args.num_samples, 1, 1)
                    
                    
                    if obs_cond.dim() == 3 and obs_cond.size(1) == 1:
                        obs_cond = obs_cond.squeeze(1)

                    # print(f"obs_cond shape for diffusion: {obs_cond.shape}")
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
                    # print(f"TIMESTEPS: {noise_scheduler.timesteps}")    
                    
                    for k in noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # print(f"SHAPES: naction {naction.shape}, noise_pred {noise_pred.shape}, timestep {k}, obs_cond {obs_cond.shape}")
                        
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

# ------------------

        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE
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
        description="Code to run nomad onnx navigation given a topomap and robot config"
    )

    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="sim_test",
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
    main(args)
