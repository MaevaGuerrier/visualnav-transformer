from vint_train.models.vint.vint import ViNT
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO
import matplotlib.pyplot as plt

from typing import Optional
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from sensor_msgs.msg import Image

from PIL import Image as PILImage
import os
import io
import time 

import numpy as np


import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    
    if model_type == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif model_type == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # checkpoint = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model



def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels

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
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    print(f"{xy}", {xy.shape})
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        cx: principal point x coordinate
        cy: principal point y coordinate
    Returns:
        camera matrix
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])







def plot_trajs_and_points_on_image(
    img: np.ndarray,
    list_trajs: list,
    pub: bool = False,
):
    """
    Plot trajectories and points on an image. If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
    Args:
        ax: matplotlib axis
        img: image to plot
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
    """
    # assert len(list_trajs) <= len(traj_colors), "Not enough colors for trajectories"
    # assert len(list_points) <= len(point_colors), "Not enough colors for points"
    # assert (
    #     dataset_name in data_config
    # ), f"Dataset {dataset_name} not found in data/data_config.yaml"

    

    # TODO MAKE IT A VARIABLE OR A CONFIG FILE
    fig, ax = plt.subplots()
    print(f"img shape: {img.shape}")
    # ax.imshow(img)
    # camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
    # camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]
    
    # camera_height =  0.05 # meters
    # camera_x_offset = 0.05 # distance between the center of the robot and the forward facing camera

    # # fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
    # # fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
    # # cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
    # # cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
    # fx, fy, cx, cy = 262.45, 263.41, 327.69, 224.45
    # camera_matrix = gen_camera_matrix(fx, fy, cx, cy)

    # # k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
    # # k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
    # # p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
    # # p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
    # # k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
    # k1, k2, p1, p2, k3 = -0.03727222045233312, 0.007588870705292973, -0.01666117486022043, 0.00581938967971292, 0.0
    # dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])
    camera_height= 0.25 # meters
    camera_x_offset= 0.2 # distance between the center of the robot and the forward facing camera
    
    fx= 272.547000
    fy= 266.358000
    cx= 320.000000
    cy= 220.000000
    
    k1= -0.038483
    k2= -0.010456
    p1= 0.003930
    p2= -0.001007
    k3= 0.0
    dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

    camera_matrix = gen_camera_matrix(fx, fy, cx, cy)


    for i, traj in enumerate(list_trajs):
        xy_coords = traj  # (horizon, 2)
        xy_coords = np.array(xy_coords)
        print(f"shape: {xy_coords.shape}")
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
        )
        if len(traj_pixels.shape) == 2:
            ax.plot(
                traj_pixels[:250, 0],
                traj_pixels[:250, 1],
                lw=2.5,
            )

    # for i, point in enumerate(list_points):
    #     if len(point.shape) == 1:
    #         # add a dimension to the front of point
    #         point = point[None, :2]
    #     else:
    #         point = point[:, :2]
    #     pt_pixels = get_pos_pixels(
    #         point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=True
    #     )
    #     ax.plot(
    #         pt_pixels[:250, 0],
    #         pt_pixels[:250, 1],
    #         color=point_colors[i],
    #         marker="o",
    #         markersize=10.0,
    #     )
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
    # ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))


    output_dir = "../debug_viz/"
    os.makedirs(output_dir, exist_ok=True)
    time_stamp = time.time()
    plt.savefig(os.path.join(output_dir, f"wps_img_{time_stamp}.png"))

    # TODO SEE IF FAST
    # Save the figure to a bytes buffer
    # buf = io.BytesIO()
    # fig.savefig(buf, format="png", bbox_inches="tight")
    # buf.seek(0)

    # Open with PIL
    # img = PILImage.open(buf)

    # time_stamp = time.time()
    # # Optionally save to file
    # output_dir = "../debug_viz/"
    # os.makedirs(output_dir, exist_ok=True)

    # return img
    # img.save(os.path.join(output_dir, f"wps_img_{time_stamp}.png"))



# # DEBUGING 


# import rospy
# import numpy as np
# import cv2
# from sensor_msgs.msg import Image

# # Example camera intrinsics from your data

# VIZ_IMAGE_SIZE = (640, 480)  # width, height

# fx, fy = 262.4592858300215, 263.41990773924925
# cx, cy = 327.6999606754441, 224.45937199538153
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0,  0,  1]])

# # Distortion coefficients
# D = np.array([-0.03727222045233312, 0.007588870705292973,
#               -0.01666117486022043, 0.00581938967971292, 0.0, 0.0, 0.0, 0.0])


# def project_points_2d(points):
#     """
#     points: Nx2 array of 2D points in camera frame (X,Y) at Z=1 (flat plane)
#     Returns: Nx2 array of pixel coordinates (u,v)
#     """
#     # Assuming Z=1 plane in front of camera
#     points_3d = np.hstack([points, np.ones((points.shape[0],1))])  # Nx3

#     # Project using K
#     uvw = (K @ points_3d.T).T  # Nx3
#     u = uvw[:,0] / uvw[:,2]
#     v = uvw[:,1] / uvw[:,2]

#     pixels = np.stack([u, v], axis=1)

#     # Clip to image bounds
#     pixels[:,0] = np.clip(pixels[:,0], 0, VIZ_IMAGE_SIZE[0]-1)
#     pixels[:,1] = np.clip(pixels[:,1], 0, VIZ_IMAGE_SIZE[1]-1)

#     return pixels.astype(np.int32)

# def overlay_trajectories(img, waypoints_list, traj_colors):
    # """
    # Draw trajectories on the image
    # """
    # for traj, color in zip(waypoints_list, traj_colors):
    #     pixels = project_points_2d(traj)
    #     # Draw lines connecting the points
    #     for i in range(len(pixels)-1):
    #         pt1 = tuple(pixels[i])
    #         pt2 = tuple(pixels[i+1])
    #         cv2.line(img, pt1, pt2, color, thickness=2)
    #     # Draw circles at waypoints
    #     for pt in pixels:
    #         cv2.circle(img, tuple(pt), 4, color, -1)
    # return img