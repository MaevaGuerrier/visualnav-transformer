#!/usr/bin/env python3
# ros node for DepthAnythingV2 depth estimation
import argparse
import os
import cv2
import numpy as np
import torch
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2


# UTILS 

from topic_names import (IMAGE_TOPIC,
                        DEPTH_IMAGE_TOPIC)


def crop_by_height(img: np.ndarray, robot_height_ratio: float = 0.6):
    """
    Crop the RGB image with respect to the robot height.
    
    Parameters:
        img (np.ndarray): Input RGB image (H, W, 3).
        robot_height_ratio (float): Ratio of the image height to keep.
                                    E.g., 0.6 -> keep bottom 60% of the image.
    Returns:
        np.ndarray: Cropped image.
    """
    h, w = img.shape[:2]
    start_y = int(h * (1 - robot_height_ratio))  # start cropping from here
    cropped = img[start_y:h, 0:w]
    return cropped


class DepthAnythingNode:
    def __init__(self, input_topic, output_topic, encoder, input_size, grayscale):
        self.bridge = CvBridge()
        self.output_topic = output_topic
        self.input_size = input_size
        self.grayscale = grayscale

        # Setup ROS pubs/subs
        self.sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher(DEPTH_IMAGE_TOPIC, Image, queue_size=1)

        # Choose device
        self.device = 'cuda' if torch.cuda.is_available() else \
                      'mps' if torch.backends.mps.is_available() else 'cpu'

        # Model configs
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        # Load model
        rospy.loginfo(f"Loading DepthAnythingV2 with encoder={encoder} on {self.device}")
        self.depth_anything = DepthAnythingV2(**model_configs[encoder])
        ckpt_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        self.depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()

        # Colormap
        self.cmap = matplotlib.colormaps['Spectral_r']

    def image_callback(self, msg: Image):
        try:
            # Convert ROS -> OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Crop
            cropped = crop_by_height(cv_img, robot_height_ratio=0.6)

            # Depth inference
            depth = self.depth_anything.infer_image(cropped, self.input_size)

            # Normalize to 0-255
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if self.grayscale:
                depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth_vis = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            # Convert back to ROS Image and publish
            depth_msg = self.bridge.cv2_to_imgmsg(depth_vis, encoding="bgr8")
            depth_msg.header = msg.header
            self.pub.publish(depth_msg)

        except Exception as e:
            rospy.logerr(f"DepthAnything processing failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DepthAnythingV2 ROS Node")
    parser.add_argument("--input-topic", type=str, default="/camera/color/image_raw")
    parser.add_argument("--output-topic", type=str, default="/depth_anything/depth_image")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    args = parser.parse_args()

    rospy.init_node("depth_anything_v2_node", anonymous=True)

    node = DepthAnythingNode(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        encoder=args.encoder,
        input_size=args.input_size,
        grayscale=args.grayscale,
    )

    rospy.loginfo(f"DepthAnythingV2 Node started. Listening on {args.input_topic}, publishing to {args.output_topic}")
    rospy.spin()
