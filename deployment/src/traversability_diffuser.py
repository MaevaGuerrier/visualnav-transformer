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
from typing import List, Tuple, Dict, Any, Deque
from collections import deque
import cv2
from cv_bridge import CvBridge
import time

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
import ros_numpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# torch
import torch
import torch.nn.functional as F

# Utils
from topic_names import (IMAGE_TOPIC, WAYPOINT_TOPIC, REACHED_GOAL_TOPIC)                 
from utils import load_model, msg_to_pil, transform_images, to_numpy, pil_to_numpy_array

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

VIZ_IMAGE_SIZE = (224, 224)

camera_matrix = np.array([
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

def project_points(xy: np.ndarray, camera_height: float, camera_x_offset: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
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
    
    # done for cv2.fisheye.projectPoint requires float32/float64 and shape (N,1,3)
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
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    # print(pixels)
    # Flip image horizontally
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]

    return pixels

def plot_trajs_and_points_on_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
):
    """
    Plot trajectories and points on an image.
    """
    camera_height = 0.25
    camera_x_offset = 0.10

    # TODO GO BACK TO HERE SOLVE THE ISSUE
#       File "traversability_diffuser.py", line 638, in <module>
#     main(args)
#   File "traversability_diffuser.py", line 536, in main
#     _publish_overlay_image(img, cam_wp_pub, naction)
#   File "traversability_diffuser.py", line 394, in _publish_overlay_image
#     img = plot_trajs_and_points_on_image(
#   File "traversability_diffuser.py", line 186, in plot_trajs_and_points_on_image
#     xy_coords = traj[:, :2]
# IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
    for i, traj in enumerate(list_trajs):
        rospy.logdebug(f"TRAJECTORIES: {traj}")
        xy_coords = traj[:, :2]
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs
        )
        
        points = traj_pixels.astype(int).reshape(-1, 1, 2)
        # Random color for each trajectory
        color = tuple(int(x) for x in np.random.choice(range(50, 255), size=3))
        # inverting x,y axis so origin in image is down-left corner
        points[:, :, 1] = VIZ_IMAGE_SIZE[1] - 1 - points[:, :, 1]
        # Draw trajectory
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=3)

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

    # Color 
    marker.color.a = 1.0  # alpha
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0

    waypoint_viz_pub.publish(marker)










# DO NOT USE JUST FOR UNDERSTANDING

# ACTION_STATS['min'] = np.array([-2.5, -4])
# ACTION_STATS['max'] = np.array([5, 4])

# self.delta_min = from_numpy(ACTION_STATS['min']).to(self.device)
# self.delta_max = from_numpy(ACTION_STATS['max']).to(self.device)

# def _norm_delta_to_ori_trajs(trajs, delta_min=None, delta_max=None):
#     delta_tmp = (trajs + 1) / 2
#     delta_ori = delta_tmp * (delta_max - delta_min) + delta_min
#     trajs_ori = delta_ori.cumsum(dim=1)
#     return trajs_ori

# def add_robot_dim(self, world_ps):
#     tangent = world_ps[:, 1:, 0:2] - world_ps[:, :-1, 0:2]
#     tangent = tangent / torch.norm(tangent, dim=2, keepdim=True)
#     normals = tangent[:, :, [1, 0]] * torch.tensor(
#         [-1, 1], dtype=torch.float32, device=world_ps.device
#     )
#     world_ps_inflated = torch.vstack([world_ps[:, :-1, :]] * 3)
#     world_ps_inflated[:, :, 0:2] = torch.vstack(
#         [
#             world_ps[:, :-1, 0:2] + normals * self.robot_width / 2,
#             world_ps[:, :-1, 0:2],  # center
#             world_ps[:, :-1, 0:2] - normals * self.robot_width / 2,
#         ]
#     )
#     return world_ps_inflated

# def collision_cost(self, trajs, scale_factor=None):
#     if self.cost_map is None:
#         return torch.zeros(trajs.shape)
#     batch_size, num_p, _ = trajs.shape
#     trajs_ori = self._norm_delta_to_ori_trajs(trajs)
#     trajs_ori = self.add_robot_dim(trajs_ori)
#     if scale_factor is not None:
#         trajs_ori *= scale_factor
#     norm_inds, _ = self.tsdf_cost_map.Pos2Ind(trajs_ori)
#     cost_grid = self.cost_map.T.expand(trajs_ori.shape[0], 1, -1, -1)
#     oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
#     oloss_M = oloss_M.to(torch.float32)

#     loss = 0.003 * torch.sum(oloss_M, axis=1)
#     if trajs.grad is not None:
#         trajs.grad.zero_()
#     loss.backward(torch.ones_like(loss))
#     cost_list = loss[1::3]
#     generate_scale = self.generate_scale(trajs.shape[1])
#     return generate_scale.unsqueeze(1).unsqueeze(0) * trajs.grad, cost_list

# def get_gradient(self, trajs, alpha=0.3, t=None, goal_pos=None, ACTION_STATS=None, scale_factor=None):
#     trajs_in = trajs.detach().requires_grad_(True).to(self.device)
#     collision_cost, cost_list = self.collision_cost(trajs_in, scale_factor=scale_factor)
#     cost = collision_cost
#     return cost, cost_list


# I have the traversability values in input space (rgb images) 2D
# I have the trajectory projected in the input space (rgb images) 2D
# I need to evaluate the cost of the trajectory in the traversability image
# I need to compute the gradient of the cost with respect to the trajectory
# I need to use the gradient to modify the trajectory
# I need to understand how the diffusion model will react to this gradient


# For now correcting only one traj for debugging
def _get_gradient_traversability(traj, fully_traversable_value=1.0):

    if trav_img is None:
        return 
    
    # traversability image, 0..1 values
    traversability_img = torch.tensor(trav_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # F.grid_sample -> grid specifies the sampling pixel locations normalized by the input spatial dimensions. 
    # Therefore, it should have most values in the range of [-1, 1]
    # see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    traj_normalized = torch.tensor(traj.copy(), dtype=torch.float32, requires_grad=True)  # [N,2]
    rospy.logdebug(f"Trajectory before traversability correction: {traj_normalized}")
    
    # grid for sampling: [1,N,1,2]
    grid = traj_normalized.unsqueeze(0).unsqueeze(2)

    traversability_vals = F.grid_sample(traversability_img, grid, align_corners=True)
    traversability_vals = traversability_vals.squeeze()  
    rospy.logdebug(f"Traversability values along the trajectory: {traversability_vals}")

    cost = torch.sum(fully_traversable_value - traversability_vals)
    cost.backward()
    rospy.logdebug(f"Trajectory gradient after traversability correction: {traj_normalized.grad}")
    rospy.logdebug(f"Cost value: {cost.item()}")

    return traj_normalized.grad

        # with torch.no_grad():
        #     traj_normalized -= lr * traj.grad
        #     traj_normalized.grad.zero_()



# Callbacks


# 0 is untraversable and 1 is fully traversable. https://arxiv.org/pdf/2404.07110
def _callback_traversability_image(trav_img_msg: Image):
    trav_img = torch.from_numpy(ros_numpy.numpify(trav_img_msg))
    # TODO CHECK 2 Traversability images values need to remains unchanged here
    is_in_range = torch.all((trav_img >= 0) & (trav_img <= 1))
    rospy.logdebug(f"Traversability image values in range [0, 1] ( 0 is untraversable and 1 is fully traversable.): {is_in_range}")
    rospy.logdebug(f"Received traversability image of shape: {trav_img.shape}")
    rospy.logdebug(f"traversability image data type: {trav_img.dtype}\n")


# TODO TRY TO NOT HAVE THE GLOBAL
overlay_traj_img = None
def _callback_traversability_overlay_image(trav_img_msg: Image):
    overlay_traj_img = ros_numpy.numpify(trav_img_msg)

def _publish_overlay_image(img: np.ndarray, pub: rospy.Publisher, trajs: List[np.ndarray]):

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Convert RGB → BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = plot_trajs_and_points_on_image(
        img=img,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        list_trajs=trajs,
    )

    ros_img = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    # cv_img = bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
    # cv2.imwrite("../debug_viz/img_ros_test2.png", cv_img)
    # exit()
    ros_img.header.stamp = rospy.Time.now()
    ros_img.header.frame_id = "base_footprint"
    pub.publish(ros_img)



# rospy.logdebug, rospy.loginfo, rospy.logwarn, rospy.logerr and rospy.logfatal

def main(args: argparse.Namespace):
    rospy.init_node("traversability_diffusor", anonymous=True, log_level=args.log_level)

    # /wild_visual_navigation_visu_traversability_front/traversability_overlayed 
    rospy.Subscriber("/wild_visual_navigation_node/front/traversability", Image, _callback_traversability_image) 
    rospy.Subscriber("/wild_visual_navigation_visu_traversability_front/traversability_overlayed", Image, _callback_traversability_overlay_image)

    # PUBLISHERS
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1) 
    goal_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1)
    chosen_wp_viz_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    all_path_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
    # OVERLAY IMAGE
    # TODO BETTER NAMING TO UNDERSTAND
    cam_wp_pub = rospy.Publisher("/topoplan/wps_overlay_img", Image, queue_size=10) # not corrected action
    cam_corr_wp_pub = rospy.Publisher("/topoplan/wps_corrected_overlay_img", Image, queue_size=10)
    trav_wp_pub = rospy.Publisher("/topoplan/wps_overlay_trav_img", Image, queue_size=10) # not corrected action
    trav_corr_wp_pub = rospy.Publisher("/topoplan/wps_corrected_overlay_trav_img", Image, queue_size=10)

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
            naction = naction[0] 
            chosen_waypoint = naction[args.waypoint]   
            rospy.logdebug(f"Chosen waypoint: {chosen_waypoint}")             

                    # interval = 6
                    # if timestep <= interval:
                    #     grad = pathguide.get_gradient(naction, goal_pos=rela_pos, scale_factor=scale_factor)
                    #     naction -= grad


            if model_params["normalize"]:
                chosen_waypoint[:2] *= (MAX_V / RATE)  
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint
            waypoint_pub.publish(waypoint_msg)
            rospy.logdebug(f"Published waypoint: {chosen_waypoint}")
            viz_chosen_wp(chosen_waypoint, chosen_wp_viz_pub)

            img = context_queue[-1]
            img = pil_to_numpy_array(image_input=img, target_size=VIZ_IMAGE_SIZE)
            _publish_overlay_image(img, cam_wp_pub, naction)
            # _publish_overlay_image(img, cam_corr_wp_pub, naction_corr)
            if overlay_traj_img is not None:
                # _publish_overlay_image(overlay_traj_img, trav_corr_wp_pub, naction_corr) # SHOULD BE THE CORRECTED ACTION SO WE CAN COMPARE EASILY
                _publish_overlay_image(trav_wp_pub, trav_corr_wp_pub, naction) # ORIG ACTION WITHOUT CORRECTION


            ma = MarkerArray()
            for idx, paths in enumerate(naction):
                r = 0.0
                g = 0.0
                b = 1.0
                marker = make_path_marker(
                    paths, idx, r, g, b, frame_id="base_link")
                ma.markers.append(marker)
            all_path_pub.publish(ma) 

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
