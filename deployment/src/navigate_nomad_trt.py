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
from std_msgs.msg import Bool, Float32MultiArray, Int32, Float32
from nav_msgs.msg import Path
from utils_onnx import msg_to_pil, transform_images, load_model_trt, load_model_onnx


# To DELETE AS WE CHECK THAT EACH TRT MODULE WORKS CORRECTLY -----------------------------
from utils import load_model, to_numpy
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from ddpms_scheduler_np import DDPMSchedulerNumPy

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

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )   

    # -------------------------------------------------------------------------------------------------------------

    # trt_vision_encoder = load_model_trt("nomad_vision_encoder.trt")
    trt_vis_encoder = load_model_trt("nomad_vision_encoder.trt")
    print("loaded vision encoder trt model")
    trt_dist_pred = load_model_trt("nomad_dist_pred_net.trt")
    print("loaded distance predictor trt model")
    ort_sess_noise_pred = load_model_onnx("nomad_noise_pred_net")
    print("loaded noise predictor onnx model")
    # load topomap
    topomap_filenames = sorted(
        os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, args.dir)),
        key=lambda x: int(x.split(".")[0]),
    )
    # print("loaded topomap images")
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
    distances_pub = rospy.Publisher("/distances", Float32MultiArray, queue_size=1)
    inference_pub = rospy.Publisher("/inference_time", Float32, queue_size=10)

    # navigation loop
    # print("befre while loop")
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        # print("in ros")
        # print("context_queue length:", len(context_queue))
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
                # print("init context_queue")
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                # batch_obs_imgs_np = []
                # batch_goal_data_np = []

                crop = False
                
                start_time = time.time()
                # Transform observation once
                transf_obs_img = transform_images(
                    context_queue, model_params["image_size"], center_crop=crop
                )

                # Vectorized goal processing
                goal_imgs = topomap[start:end + 1]  
                batch_goal_data_np = np.concatenate([
                    transform_images(sg_img, model_params["image_size"], center_crop=crop)
                    for sg_img in goal_imgs
                ], axis=0).astype('float32')

                # Repeat observation for batch
                num_goals = len(goal_imgs)
                batch_obs_imgs_np = np.tile(transf_obs_img, (num_goals, 1, 1, 1)).astype('float32')
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
                # print(f"Inference time without torch {time.time() - start_time}")
                
                # distances, waypoints = ort_outputs[0], ort_outputs[1]
                # goal_image shape: torch.Size([4, 3, 96, 96]), 
                # obs image shape: torch.Size([4, 12, 96, 96]), 
                # mask shape: torch.Size([4])
                
                # print(f"Shape batch_obs_imgs_np {batch_obs_imgs_np.shape} goal {batch_goal_data_np.shape} mask {input_goal_mask_np.shape}")
                obsgoal_cond = trt_vis_encoder.infer(obs_img=batch_obs_imgs_np, goal_img=batch_goal_data_np, input_goal_mask=input_goal_mask_np)
                # print(f"obscond {obsgoal_cond}")
                obsgoal_cond = obsgoal_cond[0]
                # print(f"obscond after {obsgoal_cond}")
                distances = trt_dist_pred.infer(obsgoal_cond=obsgoal_cond)
                # print("distances before:", distances)
                distances = distances[0]
                distances_msg = Float32MultiArray()
                distances_msg.data = distances
                distances_pub.publish(distances_msg)
                # print("distances after:", distances)
                min_dist_idx = np.argmin(distances)
                
                
               
# -----------------

                closest_node = min_dist_idx + start
                print("closest node:", closest_node)
                closest_node_msg = Int32()
                closest_node_msg.data = closest_node
                closest_node_pub.publish(closest_node_msg)
                
                sg_idx = min(min_dist_idx + int(distances[min_dist_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond_np = obsgoal_cond[sg_idx]
        
                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond_np.shape) == 2:
                        obs_cond_np = np.tile(obs_cond_np, (args.num_samples, 1))
                    else:
                        obs_cond_np = np.tile(obs_cond_np, (args.num_samples, 1, 1))

                    # we need eq. global_cond torch.Size([8, 256])
                    if obs_cond_np.ndim == 3 and obs_cond_np.shape[1] == 1:
                        obs_cond_np = obs_cond_np.squeeze(1)

                    
                    # initialize action from Gaussian noise
                    naction_np = np.random.randn(
                        args.num_samples, model_params["len_traj_pred"], 2
                    ).astype(np.float32)

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

        
                    for k in noise_scheduler.timesteps[:]:
                        # predict noise
                        k_np = np.array(k.cpu().item(), dtype=np.int64)
                        # print(f"Shape obs_cond_np: {obs_cond_np.shape}, naction_np: {naction_np.shape}, k_np: {k_np.shape}")
                        # print(f"Type obs_cond_np: {type(obs_cond_np)}, naction_np: {type(naction_np)}, k_np: {type(k_np)}")
                        # print("before ort sess noise pred")
                        ort_sess_noise_pred_inputs = {
                            "sample": naction_np,   
                            "timestep": k_np,
                            "global_cond": obs_cond_np,
                        }
                        noise_pred = ort_sess_noise_pred.run(None, ort_sess_noise_pred_inputs)[0]
                        # print("after ort sess noise pred")
                        # naction shape (8, 8, 2) type <class 'numpy.ndarray'>, noise_pred shape (8, 8, 2) type <class 'numpy.ndarray'>, k 9 type <class 'int'>
                        # inverse diffusion step (remove noise)
                        # DDPMScheduler need torch tensors (@TODO find a numpy implementation?)
                        noise_pred_torch = torch.from_numpy(noise_pred).float().to(device)
                        naction_torch = torch.from_numpy(naction_np).float().to(device)
                        # print("before noise scheduler")
                        naction_torch = noise_scheduler.step(
                            model_output=noise_pred_torch,
                            timestep=k,
                            sample=naction_torch
                        ).prev_sample
                        # print(f"After noise scheduler")
                        naction_np = naction_torch.detach().cpu().numpy()
                        # print(f"naction after noise scheduler {naction_np}")


                    inference_time = time.time() - start_time
                    print(f"Inference time: {inference_time:.3f} seconds")
                    inference_time_msg = Float32()
                    inference_time_msg.data = inference_time
                    inference_pub.publish(inference_time_msg)
                    
                naction_torch = torch.from_numpy(naction_np).float().to(device)
                naction_np = to_numpy(get_action(naction_torch))
                # naction_np = get_action(naction_np)
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction_np.flatten()))
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)
                naction_np = naction_np[0] 
                chosen_waypoint = naction_np[args.waypoint]

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
