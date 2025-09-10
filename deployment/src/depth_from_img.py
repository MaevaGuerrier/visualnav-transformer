#!/usr/bin/env python3
# ros node for DepthAnythingV2 depth estimation
import argparse
import os
import cv2
import numpy as np
import torch
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2


# UTILS 

from topic_names import (IMAGE_TOPIC,
                        DEPTH_IMAGE_UNDISTORTED_TOPIC,
                        IMAGE_UNDISTORTED_TOPIC,
                        CAMERA_INFO_TOPIC)


# def crop_by_height(img: np.ndarray, robot_height_ratio: float = 0.6):
#     """
#     Crop the RGB image with respect to the robot height.
    
#     Parameters:
#         img (np.ndarray): Input RGB image (H, W, 3).
#         robot_height_ratio (float): Ratio of the image height to keep.
#                                     E.g., 0.6 -> keep bottom 60% of the image.
#     Returns:
#         np.ndarray: Cropped image.
#     """
#     h, w = img.shape[:2]
#     start_y = int(h * (1 - robot_height_ratio))  # start cropping from here
#     cropped = img[start_y:h, 0:w]
#     return cropped


class DepthAnythingNode:
    def __init__(self, encoder, input_size):
        self.bridge = CvBridge()
        self.input_size = input_size

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
        # TODO MAKE IT DYNAMIC OR MAKE ALL SUBMODULE IN SINGLE FOLDER
        ckpt_path = f'../../../../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'
        self.depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()

        # Colormap
        self.cmap = matplotlib.colormaps['Spectral_r']

        # TODO ALL CAM INFO WILL NEED TO BE IN A CONFIG FILE
        # fisheye info, not using cam_info in case not populated
        self.camera_matrix = np.array([
            [262.459286,   1.916160, 327.699961],
            [  0.000000, 263.419908, 224.459372],
            [  0.000000,   0.000000,   1.000000]
        ], dtype=np.float64)

        self.distortion_coeffs = np.array([
            -0.03727222045233312, 
                0.007588870705292973,
            -0.01666117486022043, 
                0.00581938967971292
        ], dtype=np.float64)

        # Setup ROS pubs/subs
        self.sub_img = rospy.Subscriber(IMAGE_TOPIC, Image, self.rgb_depth_callback, queue_size=10, buff_size=2**24)
        # self.sub_bird_eye = rospy.Subscriber(DEPTH_IMAGE_TOPIC, Image, self.bird_eye_callback, queue_size=1, buff_size=2**24)
        self.pub_undistort_depth = rospy.Publisher(DEPTH_IMAGE_UNDISTORTED_TOPIC, Image, queue_size=10)
        self.pub_undistort_img = rospy.Publisher(IMAGE_UNDISTORTED_TOPIC, Image, queue_size=10)
        # self.pub_bird_eye_depth = rospy.Publisher(BIRD_EYE_DEPTH_TOPIC, Image, queue_size=1)
        self.pub_cam_info = rospy.Publisher(CAMERA_INFO_TOPIC, CameraInfo, queue_size=10)


    # def depth_to_birds_eye(self, depth_image, camera_matrix, distortion_coeffs, output_size=(500, 500), scale=100):
    #     """
    #     Convert a distorted fisheye depth image into a bird's-eye view projection.

    #     Args:
    #         depth_image (np.ndarray): Input distorted depth image (H x W).
    #         camera_matrix (np.ndarray): 3x3 intrinsic camera matrix.
    #         distortion_coeffs (np.ndarray): Distortion coefficients (fisheye model).
    #         output_size (tuple): Size of bird’s-eye occupancy grid (width, height).
    #         scale (float): Scaling factor for converting world meters -> grid pixels.

    #     Returns:
    #         birds_eye_map (np.ndarray): Bird's-eye view occupancy grid.
    #     """
    #     height, width = depth_image.shape # H, W

    #     # Step 1: Undistort depth image (for fisheye camera)
    #     new_camera_matrix = camera_matrix.copy()
    #     map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
    #         camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (width, height), cv2.CV_16SC2#CV_32FC1
    #     )
    #     undistorted_depth = cv2.remap(depth_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    #     if self.debug:
    #         self.pub_undistorted_depth.publish(self.bridge.cv2_to_imgmsg(undistorted_depth))

    #     # Step 2: Generate pixel grid
    #     pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    #     pixel_homog = np.stack((pixel_x, pixel_y, np.ones_like(pixel_x)), axis=-1).reshape(-1, 3).T  # (3, N)

    #     # Step 3: Back-project pixels into 3D camera coordinates
    #     inv_camera_matrix = np.linalg.inv(camera_matrix)
    #     rays = inv_camera_matrix @ pixel_homog  # shape (3, N)
    #     depth_values = undistorted_depth.reshape(-1)
    #     points_3d = rays * depth_values  # scale rays by depth

    #     # Step 4: Keep only ground plane points (z > 0)
    #     x_world = points_3d[0, :]
    #     y_world = points_3d[1, :]
    #     z_world = points_3d[2, :]
    #     valid_indices = (z_world > 0) & (depth_values > 0)

    #     x_world = x_world[valid_indices]
    #     y_world = y_world[valid_indices]
    #     z_world = z_world[valid_indices]

    #     # Step 5: Project to bird's-eye (X,Z plane -> grid map)
    #     grid_width, grid_height = output_size
    #     birds_eye_map = np.zeros((grid_height, grid_width), dtype=np.uint8)

    #     grid_x = (x_world * scale + grid_width // 2).astype(int)
    #     grid_y = (z_world * scale).astype(int)

    #     valid_mask = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
    #     birds_eye_map[grid_y[valid_mask], grid_x[valid_mask]] = 255  # mark as occupied

    #     return birds_eye_map

    def publish_camera_info(self, camera_matrix, width, height, header):
        msg = CameraInfo()
        # msg.header = header# Change if your camera frame has a different name
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id

        msg.width = width
        msg.height = height

        # Fill intrinsic matrices
        msg.K = camera_matrix.flatten().tolist()   # 3x3 row-major

        msg.distortion_model = "plumb_bob"
        msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]  # no fisheye distortion

        msg.R = np.eye(3).flatten().tolist()                # rectification = identity
        # Projection matrix (3x4) → [K | 0]
        P = np.zeros((3, 4))
        P[:3, :3] = camera_matrix
        msg.P = P.flatten().tolist()
        
        # print(f"TIME IN MSG INFO: {msg.header.stamp}")
        self.pub_cam_info.publish(msg)

    def rgb_depth_callback(self, msg: Image):
        try:

            img_cv2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height_img, width_img, _ = img_cv2.shape # H, W
            # print(f"Received image of size: {width_img}x{height_img}")

            # Step 1: Undistort image (for fisheye camera)
            new_camera_matrix = self.camera_matrix.copy()
            # TODO PUT BACK ITS FOR TEST WITH OAK PRO
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.distortion_coeffs, np.eye(3), new_camera_matrix, (width_img, height_img), cv2.CV_16SC2#CV_32FC1 TODO Look that up
            )
            self.publish_camera_info(new_camera_matrix, width_img, height_img, msg.header)
            img_undistort = cv2.remap(img_cv2, map_x, map_y, interpolation=cv2.INTER_LINEAR)

            # Depth inference
            # IMAGE UNDISTORT PUT IT BACK IF USING FISHEYE
            # img_undistort = img_cv2 # use for oak
            depth = self.depth_anything.infer_image(img_undistort, self.input_size)

            # # Normalize to 0-255
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            # Convert back to ROS Image and publish
            # print(f"TIME IN RGB DEPT: {msg.header.stamp}")
            img_msg = self.bridge.cv2_to_imgmsg(img_undistort)
            img_msg.header.stamp = msg.header.stamp
            img_msg.header.frame_id = msg.header.frame_id

            depth_msg = self.bridge.cv2_to_imgmsg(depth)
            depth_msg.header.stamp = msg.header.stamp
            depth_msg.header.frame_id = msg.header.frame_id


            # self.pub_undistort_depth.publish(depth_msg)
            self.pub_undistort_img.publish(img_msg)

        except Exception as e:
            rospy.logerr(f"DepthAnything processing failed: {e}")



    # def bird_eye_callback(self, msg: Image):
    #     cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # depth image should not become rgb
    #     img = self.depth_to_birds_eye(cv_img, self.camera_matrix, self.distortion_coeffs)
    #     # Publish the bird's-eye view image
    #     bird_eye_msg = self.bridge.cv2_to_imgmsg(img)
    #     self.pub_bird_eye_depth.publish(bird_eye_msg)

# TODO CAMERA INFO PUBLISH 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DepthAnythingV2 ROS Node")
    # TODO just make a var
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()

    rospy.init_node("depth_anything_v2_node", anonymous=True)

    node = DepthAnythingNode(
        encoder=args.encoder,
        input_size=args.input_size
    )

    rospy.loginfo(f"DepthAnythingV2 Node started.")
    rospy.spin()
