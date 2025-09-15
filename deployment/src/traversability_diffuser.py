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
# ROS
import rospy
from sensor_msgs.msg import Image
import ros_numpy
# torch
import torch



# Constants variables
## Paths
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
## Robot parameters
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 


# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)



# UTILS FUNCTIONS TODO ONCE DONE PUT THEM IN UTILS FILE

# def _load_model(model_name: str, device: torch.device, mode: str = "eval"):
#     with open(MODEL_CONFIG_PATH, "r") as f:
#         model_paths = yaml.safe_load(f)

#     model_conf_path = model_paths[model_name]["config_path"]

#     ckpt_path = model_paths[model_name]["ckpt_path"]
#     with open(model_conf_path, "r") as f:
#         model_params = yaml.safe_load(f)

#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

#     print(f"Loading model from {ckpt_path}")
#     model = load_model(ckpt_path, model_params, device).to(device).eval()
#     return model, model_params




# CALLBACKS


# 0 is untraversable and 1 is fully traversable. https://arxiv.org/pdf/2404.07110
def callback_traversability_image(trav_img_msg: Image):
    trav_img = ros_numpy.numpify(trav_img_msg)
    rospy.logdebug(f"Received traversability image of shape: {trav_img.shape}")
    rospy.logdebug(f"traversability image data type: {trav_img.dtype}")


def main(args: argparse.Namespace):
    rospy.init_node("traversability_diffusor", anonymous=True, log_level=args.log_level)

    rospy.Subscriber("/wild_visual_navigation_node/front/traversability", Image, callback_traversability_image) 

    
    rate = rospy.Rate(RATE)  

    while not rospy.is_shutdown():




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
