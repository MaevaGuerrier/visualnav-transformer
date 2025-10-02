import numpy as np
from typing import Tuple
from viz_utils import get_pos_pixels
import torch
import torch.nn.functional as F
from threading import Event

# ROS
import ros_numpy
import rospy
from sensor_msgs.msg import Image


from utils import msg_to_pil, to_numpy
from vint_train.training.train_utils import get_action

def get_traj_pixels_coords(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
    viz_img_size: Tuple[int, int],
    resize_factor:bool=True):
    
    traj_pix_coords = []
   
    camera_height = 0.25
    camera_x_offset = 0.10

    for traj in list_trajs:
        # print("traj shape:", traj.shape)
        # print("traj:", traj)
        traj = np.asarray(traj)
        xy_coords = traj[:, :2]
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, viz_img_size
        )
        
        if resize_factor: # Traversability image is 224 x 224 and the original fisheye image is 640 x 480
            traj_pixels[:,0] *= .35
            traj_pixels[:,1] *= .46

        points = traj_pixels.astype(int).reshape(-1, 1, 2)

        # inverting x,y axis so origin in image is down-left corner
        if resize_factor:
            points[:, :, 1] = viz_img_size[1] * .46  - 1 - points[:, :, 1]
        else:
            points[:, :, 1] = viz_img_size[1] - 1 - points[:, :, 1]

        # Draw trajectory
        traj_pix_coords.append(points)

    return traj_pix_coords


def select_traj_best_traversability(
    trav_img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
    viz_img_size: Tuple[int, int],
    resize_factor:bool=True
):
    """
    Select the trajectory with the best traversability score.
    trav_img: 2D numpy array with traversability values in range [0, 1] (0 = untraversable, 1 = fully traversable)
    """
    # TODO there might be a cleaner way to do this
    if trav_img is None:
        return None, None
    
    traj_pix_coords = get_traj_pixels_coords(
        camera_matrix, dist_coeffs, list_trajs, viz_img_size, resize_factor=resize_factor
    )

    best_traj = None
    best_score = -np.inf

    for traj, pix_coords in zip(list_trajs, traj_pix_coords):
        score = 0.0
        for point in pix_coords:
            x, y = point[0]
            score += trav_img[y, x] 
        rospy.logdebug(f"Trajectory score: {score}")
        if score > best_score:
            best_score = score
            best_traj = traj

    return best_traj, best_score


class traversabilityImageSubscriber:
    def __init__(self):
        self.trav_img = None
        self.overlay_traj_img = None
        self._trav_img_ready = Event() 

        rospy.Subscriber("/wild_visual_navigation_node/front/traversability", Image, self._callback_traversability_image, queue_size=10) 
        rospy.Subscriber("/wild_visual_navigation_visu_traversability_front/traversability_overlayed", Image, self._callback_traversability_overlay_image, queue_size=10)

        rospy.loginfo("Waiting for traversability image to be processed at least once...")
        self._trav_img_ready.wait()


    # 0 is untraversable and 1 is fully traversable. https://arxiv.org/pdf/2404.07110
    def _callback_traversability_image(self, trav_img_msg: Image):
        self.trav_img = ros_numpy.numpify(trav_img_msg)
        is_in_range = np.all((self.trav_img >= 0) & (self.trav_img <= 1))
        assert is_in_range, "Traversability image values are out of range [0, 1]"
        self._trav_img_ready.set()
        rospy.logdebug(f"Received traversability image of shape: {self.trav_img.shape}")
        rospy.logdebug(f"Traversability image data type: {self.trav_img.dtype}\n")

    def _callback_traversability_overlay_image(self, trav_img_msg: Image):
        self.overlay_traj_img = ros_numpy.numpify(trav_img_msg)
        rospy.logdebug(f"Received traversability overlay image of shape: {self.overlay_traj_img.shape}")

    def get_trav_img(self):
        return self.trav_img

    def get_overlay_traj_img(self):
        return self.overlay_traj_img




