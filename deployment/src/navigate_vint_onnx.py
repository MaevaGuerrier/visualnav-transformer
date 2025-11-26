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
from utils_onnx import msg_to_pil, transform_images, load_model_onnx

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


def main(args: argparse.Namespace):
    global context_size

    vint_onnx = load_model_onnx("vint")
    print("loaded model")
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
    distances_pub = rospy.Publisher("/distances", Float32MultiArray, queue_size=1)
    inference_pub = rospy.Publisher("/inference_time", Float32, queue_size=10)

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

                crop = True
                
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


                distances, waypoints = vint_onnx.run(None, {
                    "obs_img": batch_obs_imgs_np,
                    "goal_img": batch_goal_data_np,
                })
                inference_time = time.time() - start_time
                print(f"Inference time: {inference_time:.3f} seconds")
                inference_time_msg = Float32()
                inference_time_msg.data = inference_time
                inference_pub.publish(inference_time_msg)

                distances_msg = Float32MultiArray()
                distances_msg.data = distances.flatten()
                distances_pub.publish(distances_msg)
                
                
                # print("distances shape:", distances.shape, "len:", distances)
                # print("waypoints shape:", waypoints.shape, "len:", waypoints)

                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                # print(
                #     "min dist idx:",
                #     min_dist_idx,
                #     "min dist:",
                #     distances[min_dist_idx],
                #     "close_threshold:",
                #     args.close_threshold,
                # )
                if distances[min_dist_idx] > args.close_threshold:
                    # print(
                    #     "Not close enough to the next node, choosing closest waypoint",
                    #     waypoints[min_dist_idx][args.waypoint],
                    #     "at index",
                    #     min_dist_idx,
                    # )
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    # print(
                    #     "Very far already a lost cause ",
                    #     min(min_dist_idx + 1, len(waypoints) - 1),
                    # )
                    chosen_waypoint = waypoints[
                        min(min_dist_idx + 1, len(waypoints) - 1)
                    ][args.waypoint]
                    # print(
                    #     "closest start",
                    #     start,
                    #     "min_dist_idx + 1",
                    #     min_dist_idx + 1,
                    #     "goal_node",
                    #     goal_node,
                    # )
                    closest_node = min(start + min_dist_idx + 1, goal_node)
                # print("chosen wp", chosen_waypoint)
                # print("min dist idx", min_dist_idx)

                print("closest node:", closest_node)
                closest_node_msg = Int32()
                closest_node_msg.data = closest_node
                closest_node_pub.publish(closest_node_msg)

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
                # print("------")
                for wp in waypoints[min_dist_idx]:
                    # print("waypoint:", wp)
                    path_msg_viz.poses.append(
                        PoseStamped(pose=Pose(position=Point(x=wp[0], y=wp[1])))
                    )
                path_viz_pub.publish(path_msg_viz)
                # for dist in distances:
                # print("distance:", dist)

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
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (only vint is supported) (hint: check ../config/models.yaml) (default: vint)",
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
        default="mist_office",
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
