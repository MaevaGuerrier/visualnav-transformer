import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from cv_bridge import CvBridge
import cv2



import matplotlib.pyplot as plt
import yaml

# ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import Path
from utils import msg_to_pil, to_numpy, transform_images, load_model, pil_to_msg, pil_to_numpy_array
# from vint_utils import plot_trajs_and_points_on_image

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
from visualization_msgs.msg import Marker, MarkerArray

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)


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

# GLOBALS
context_queue = []
context_size = None  
subgoal = []

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

bridge = CvBridge()

# TODO 

VIZ_IMAGE_SIZE = (640, 480)

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
    clip: bool = False,
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

    for i, traj in enumerate(list_trajs):
        xy_coords = traj[:, :2]
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
        )
        
        
        points = traj_pixels.astype(int).reshape(-1, 1, 2)
        # print(f"points shape {points.shape}, traj_pixels shape {traj_pixels.shape}")
        # print(points[0])
        # print(points[:, :, ::-1][0])
        # points = points[:, :, ::-1]
        # Random color for each trajectory
        color = tuple(int(x) for x in np.random.choice(range(50, 255), size=3))

        # inverting x,y axis so origin in image is down-left corner
        points[:, :, 1] = VIZ_IMAGE_SIZE[1] - 1 - points[:, :, 1]

        # Draw trajectory
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=3)

        # Draw start point (green) and goal point (red)
        # start = tuple(points[0, 0])
        # goal = tuple(points[-1, 0])
        # cv2.circle(img, start, 6, (0, 255, 0), -1)
        # cv2.circle(img, goal, 6, (0, 0, 255), -1)

    return img



def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    # obs_img.save("../debug_viz/obs_img.png")
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

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


def main(args: argparse.Namespace):
    global context_size
    # load model parameters

    fig, ax = plt.subplots()


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
        "viz_path", Path, queue_size=10)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    goal_img_pub = rospy.Publisher("/topoplan/goal_img", Image, queue_size=1)
    subgoal_img_pub = rospy.Publisher("/topoplan/subgoal_img", Image, queue_size=1)
    closest_node_img_pub = rospy.Publisher("/topoplan/closest_node_img", Image, queue_size=1)
    chosen_wp_viz_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    all_path_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
    cam_wp_viz_pub = rospy.Publisher("/topoplan/wps_overlay_img", Image, queue_size=10)
    # fancy_camera_pub = rospy.Publisher("/textured_quad", TexturedQuad, queue_size=1)


    # print("Registered with master node. Waiting for image observations...")

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
                # print(f"shape of naction: {naction.shape}, naction: {naction}")
                naction_selected = naction[0] 
                # print(f"batch reduce? {naction}")

                # # TODO CLEANUP MAKE A FUNC OR SMT
                # save images 
                # cv2.imwrite(f"../debug/obs_img.png", pil_to_numpy_array(context_queue[-1], target_size=(224,224)))
                # print(naction[0:2])
                # exit()
                


                chosen_waypoint = naction_selected[args.waypoint]
                print(f"WAYPOINT WE VIZ VALUE {chosen_waypoint}")
                viz_chosen_wp(chosen_waypoint, chosen_wp_viz_pub)

                img = context_queue[-1]
                img = pil_to_numpy_array(image_input=img, target_size=VIZ_IMAGE_SIZE)

                print("Image shape:", img.shape, "dtype:", img.dtype, "min:", img.min(), "max:", img.max())

                # Ensure uint8
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)

                # Convert RGB → BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # # Draw a few obvious circles
                # h, w = img.shape[:2]
                # cv2.circle(img, (312, 224), 50, (0, 0, 255), -1)   # red
                # cv2.circle(img, (w//2, h//2), 20, (0, 255, 255), -1) # yellow in center
                # cv2.circle(img, (50, 50), 30, (255, 0, 255), -1)   # magenta in corner
                # print("---------")
                # print(naction)
                # print("---------")
                img = plot_trajs_and_points_on_image(
                    img=img,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    list_trajs=naction,
                )


                # print(naction)
                # fig, ax = plt.subplots()

                # ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
                # ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))
                # ax.imshow(img)
                # plt.savefig('../debug_viz/img_test2.png')


                # Convert back to ROS image
                ros_img = bridge.cv2_to_imgmsg(img, encoding="bgr8")
                # cv_img = bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
                # cv2.imwrite("../debug_viz/img_ros_test2.png", cv_img)
                # exit()
                ros_img.header.stamp = rospy.Time.now()
                ros_img.header.frame_id = "base_footprint"
                # ros_img.encoding = "rgb8"
                cam_wp_viz_pub.publish(ros_img)
                            
                # path_msg_viz = Path()
                # path_msg_viz.header.frame_id = "base_link"
                # path_msg_viz.header.stamp = rospy.Time.now()
                # for wp in naction:
                #     path_msg_viz.poses.append(PoseStamped(
                #         pose=Pose(position=Point(x=wp[0], y=wp[1]))))
                
                # path_viz_pub.publish(path_msg_viz)
                ma = MarkerArray()
                for idx, paths in enumerate(naction):
                    r = 0.0
                    g = 0.0
                    b = 1.0
                    marker = make_path_marker(
                        paths, idx, r, g, b, frame_id="base_link")
                    ma.markers.append(marker)
                all_path_pub.publish(ma)                


                # traj_colors = [(0,255,255), (255,0,255)] 
                # img = context_queue[-1]
                # img = pil_to_numpy_array(img.copy())
                # img_overlay = overlay_trajectories(img, naction, traj_colors)
                # ros_img = bridge.cv2_to_imgmsg(img_overlay, encoding="bgr8")
                # ros_img.header.stamp = rospy.Time.now()
                # ros_img.header.frame_id = "camera"
                # cam_wp_viz_pub.publish(ros_img)      



            else: # THIS IS NOT NOAMD SO VINT OR GNM ? Its seems its using subgoal (Vint paper talked about subgoal -> subgoal candidates)
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                
                crop=True
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"], center_crop=crop)
                    goal_data = transform_images(sg_img, model_params["image_size"], center_crop=crop)
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                goal_img = transform_images(topomap[goal_node], model_params["image_size"], center_crop=crop, return_img=True)
                goal_img_msg = pil_to_msg(goal_img)
                goal_img_msg.header.stamp = rospy.Time.now()
                goal_img_msg.header.frame_id = "base_footprint"
                goal_img_msg.encoding = "rgb8"
                goal_img_pub.publish(goal_img_msg)

                subgoal_img = transform_images(topomap[end], model_params["image_size"], center_crop=crop, return_img=True)
                subgoal_img_msg = pil_to_msg(subgoal_img)
                subgoal_img_msg.header.stamp = rospy.Time.now()
                subgoal_img_msg.header.frame_id = "base_footprint"
                subgoal_img_msg.encoding = "rgb8"
                subgoal_img_pub.publish(subgoal_img_msg)


                closest_node_img = transform_images(topomap[closest_node], model_params["image_size"], center_crop=crop, return_img=True)
                closest_node_img_msg = pil_to_msg(closest_node_img)
                closest_node_img_msg.header.stamp = rospy.Time.now()
                closest_node_img_msg.header.frame_id = "base_footprint"
                closest_node_img_msg.encoding = "rgb8"
                closest_node_img_pub.publish(closest_node_img_msg)


                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                print("batch_obs_imgs shape:", batch_obs_imgs.shape)
                print("batch_goal_data shape:", batch_goal_data.shape)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)

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
        print(f"CHOSEN WAYPOINT NORMALIZED: {chosen_waypoint}")

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
        default="bunk1_office_23April",
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

