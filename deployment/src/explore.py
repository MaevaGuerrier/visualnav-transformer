
import os
from typing import Tuple, Dict, Any
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import yaml

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from vint_train.training.train_utils import get_action
import torch
import numpy as np
import argparse
import yaml
import time

# UTILS
from utils import msg_to_pil, to_numpy, transform_images, load_model, pil_to_numpy_array
from viz_utils import publish_overlay_image, viz_chosen_wp, make_marker_array
from trav_utils import * # TODO ONCE DONE IMPORT ONLY WHAT IS NEEDED
from repulsive_test import RepulsiveFieldPlanner

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
# camera_matrix_orig = np.array([
#     [262.459286,   1.916160, 327.699961],
#     [  0.000000, 263.419908, 224.459372],
#     [  0.000000,   0.000000,   1.000000]
# ], dtype=np.float64)

# dist_coeffs = np.array([
#     -0.03727222045233312, 
#         0.007588870705292973,
#     -0.01666117486022043, 
#         0.00581938967971292
# ], dtype=np.float64)

# camera_extrinsics = np.array([[0, 0, 1, 0.000],
#                             [-1, 0, 0, 0.000],
#                             [0, -1, 0, 0.025],
#                             [0, 0, 0, 1]])



# LIMO ROS AGILEX SIMULATION
camera_matrix_orig = np.array([
                [381.36246688113556,   0.0, 320.5],
                [  0.0,               381.36246688113556, 240.5],
                [  0.0,                 0.0,   1.0]
            ])
dist_coeffs = None



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
    chosen_wp_viz_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    all_path_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)
    

    # Images overlay
    cam_wp_pub = rospy.Publisher("/wps_overlay_img", Image, queue_size=10) # not corrected action
    cam_corr_wp_pub = rospy.Publisher("wps_corrected_overlay_img", Image, queue_size=10)
    trav_wp_pub = rospy.Publisher("wps_overlay_trav_img", Image, queue_size=10) # not corrected action
    trav_corr_wp_pub = rospy.Publisher("wps_corrected_overlay_trav_img", Image, queue_size=10)

    rospy.loginfo("Waiting for image observations...")
    rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=None)

    rospy.loginfo("Waiting for traversability observations...")
    rospy.wait_for_message("/wild_visual_navigation_node/front/traversability", Image, timeout=None)
    trav_img_subscriber = traversabilityImageSubscriber()
    

    while not rospy.is_shutdown():
        # EXPLORATION MODE
        waypoint_msg = Float32MultiArray()
        if (
                len(context_queue) > model_params["context_size"]
            ):

            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=True)
            obs_images = obs_images.to(args.device)
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(args.device)
            mask = torch.ones(1).long().to(args.device) # ignore the goal

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
                    (args.num_samples, model_params["len_traj_pred"], 2), device=args.device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

            start_time = time.time()
            for k in noise_scheduler.timesteps[:]:
                with torch.no_grad():
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            print("time elapsed:", time.time() - start_time)


            naction = to_numpy(get_action(naction))

            orig_naction = naction
            

            trav_img = trav_img_subscriber.get_trav_img()

            planner = RepulsiveFieldPlanner(
                robot_radius_pixels=1, # in pixel space
                max_linear_vel=MAX_V, # m/s
                max_angular_vel=MAX_W, # rad/s
                repulsion_gain=10.0,
                safety_margin_pixels=5,
                influence_radius_pixels=1
            )


            planner.create_distance_field(trav_img, threshold=0.5)
            trajectories_pixel = get_traj_pixels_coords(
                camera_matrix_orig, dist_coeffs, list(naction), VIZ_IMAGE_SIZE_FISHEYE, resize_factor=True
            )

            trajectories_pixel = np.asarray(trajectories_pixel)
            trajectories_pixel = np.asarray(trajectories_pixel).squeeze(axis=2)
            # safe_trajs = trajectories_pixel
            # print(f"pixel traj {safe_trajs.shape}")

            safe_trajs = planner.modify_trajectory(trajectory_pixels=trajectories_pixel, traversability_map=trav_img, camera_matrix=camera_matrix_orig, dist_coeffs=dist_coeffs, viz_img_size=VIZ_IMAGE_SIZE_FISHEYE, dt=1.0/RATE)
            safe_trajs = np.array(safe_trajs).squeeze()
            # print(f"MODIFY TRAJ shape {safe_trajs.shape}")

            naction = planner.get_world_coords_from_pixels(
                pixel_coords=safe_trajs,
                camera_matrix=camera_matrix_orig,
                dist_coeffs=dist_coeffs,
                viz_img_size=VIZ_IMAGE_SIZE_FISHEYE,
                camera_height=0.25,
                camera_x_offset=0.10,
                resize_factor=True
            )

            # print("Original action:", orig_naction)
            # print("Corrected action:", naction)
            # exit()


            # TODO ADD ALL CAM INFO AS CONFIG FILE 
            naction_selected = naction[0]

            # TODO either we choose based on best traj or we let it be 
            # naction_selected = naction[0] # change this based on heuristic 


            chosen_waypoint = naction_selected[args.waypoint]
            rospy.loginfo(f"chosen waypoint {chosen_waypoint}")
            rospy.loginfo(f"chosen waypoint if orig action {orig_naction[0][args.waypoint]}")
            viz_chosen_wp(chosen_waypoint, chosen_wp_viz_pub)


            img = context_queue[-1]
            img = pil_to_numpy_array(image_input=img, target_size=VIZ_IMAGE_SIZE_FISHEYE)
            publish_overlay_image(camera_matrix_orig, dist_coeffs, img, cam_wp_pub, orig_naction, viz_img_size=VIZ_IMAGE_SIZE_FISHEYE)
            publish_overlay_image(camera_matrix_orig, dist_coeffs, img, cam_corr_wp_pub, naction, viz_img_size=VIZ_IMAGE_SIZE_FISHEYE)



            overlay_traj_img = trav_img_subscriber.get_overlay_traj_img()
            if overlay_traj_img is not None:
                publish_overlay_image(camera_matrix_orig, dist_coeffs, overlay_traj_img, trav_wp_pub, orig_naction, viz_img_size=VIZ_IMAGE_SIZE_FISHEYE, resize_factor=True) # ORIG ACTION WITHOUT CORRECTION
                publish_overlay_image(camera_matrix_orig, dist_coeffs, overlay_traj_img, trav_corr_wp_pub, naction, viz_img_size=VIZ_IMAGE_SIZE_FISHEYE, resize_factor=True) # ORIG ACTION WITHOUT CORRECTION


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
   
    
    parser.add_argument(
            "--debug", action="store_true", help="Enable debug mode with verbose logging"
        )

    parser.add_argument(
        "--enable_trav", 
        "-e",
        default=True,
        type=bool,
        help="Enable traversability guidance during diffusion (default: True)",
    )

    args = parser.parse_args()
    args.log_level = rospy.DEBUG if args.debug else rospy.INFO

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.loginfo(
        f"Log level set to: {args.log_level}\n"
        f"Using device: {args.device}\n"
        f"_____________________________________________\n"
        f"Listening to image topic {IMAGE_TOPIC} \n Publishing to topic {robot_config['vel_navi_topic']} with observation rate at {robot_config['frame_rate']} Hz"
    )

    main(args)


