import numpy as np
from typing import Tuple
from viz_utils import get_pos_pixels
# ROS
import ros_numpy
import rospy
from sensor_msgs.msg import Image

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



# TODO FIGURE IT OUT
# For now correcting only one traj for debugging
# def get_gradient_traversability(trajs, trav_img, fully_traversable_value=1.0):

#     if trav_img is None:
#         return 
    
#     # traversability image, 0..1 values
#     traversability_img = torch.tensor(trav_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

#     # F.grid_sample -> grid specifies the sampling pixel locations normalized by the input spatial dimensions. 
#     # Therefore, it should have most values in the range of [-1, 1]
#     # see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#     traj_normalized = torch.tensor(trajs.copy(), dtype=torch.float32, requires_grad=True)  # [N,2]
#     rospy.logdebug(f"Trajectory before traversability correction: {traj_normalized}")
    
#     # grid for sampling: [1,N,1,2]
#     grid = traj_normalized.unsqueeze(0).unsqueeze(2)

#     traversability_vals = F.grid_sample(traversability_img, grid, align_corners=True)
#     traversability_vals = traversability_vals.squeeze()  
#     rospy.logdebug(f"Traversability values along the trajectory: {traversability_vals}")

#     cost = torch.sum(fully_traversable_value - traversability_vals)
#     cost.backward()
#     rospy.logdebug(f"Trajectory gradient after traversability correction: {traj_normalized.grad}")
#     rospy.logdebug(f"Cost value: {cost.item()}")

#     return traj_normalized.grad

#         # with torch.no_grad():
#         #     traj_normalized -= lr * traj.grad
#         #     traj_normalized.grad.zero_()



class traversabilityImageSubscriber:
    def __init__(self):
        self.trav_img = None
        self.overlay_traj_img = None

        rospy.Subscriber("/wild_visual_navigation_node/front/traversability", Image, self._callback_traversability_image, queue_size=10) 
        rospy.Subscriber("/wild_visual_navigation_visu_traversability_front/traversability_overlayed", Image, self._callback_traversability_overlay_image, queue_size=10)


    # 0 is untraversable and 1 is fully traversable. https://arxiv.org/pdf/2404.07110
    def _callback_traversability_image(trav_img_msg: Image):

        trav_img = ros_numpy.numpify(trav_img_msg)
        is_in_range = np.all((trav_img >= 0) & (trav_img <= 1))
        rospy.logdebug(
            f"Traversability image values in range [0, 1] (0 = untraversable, 1 = fully traversable): {is_in_range}"
        )
        rospy.logdebug(f"Received traversability image of shape: {trav_img.shape}")
        rospy.logdebug(f"Traversability image data type: {trav_img.dtype}\n")

    def _callback_traversability_overlay_image(trav_img_msg: Image):
        overlay_traj_img = ros_numpy.numpify(trav_img_msg)
        rospy.logdebug(f"Received traversability overlay image of shape: {overlay_traj_img.shape}")


    def get_trav_img(self):
        return self.trav_img
    
    def get_overlay_traj_img(self):
        return self.overlay_traj_img