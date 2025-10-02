import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from typing import Tuple, List
import cv2
from cv_bridge import CvBridge


bridge = CvBridge()


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
    
    if dist_coeffs is None:

        uv, _ = cv2.projectPoints(
            xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
        )

    else:

        # done for cv2.fisheye.projectPoint requires float32/float64 and shape (N,1,3),
        xyz_cv = xyz_cv.reshape(batch_size * horizon, 1, 3).astype(np.float64)

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
    viz_img_size: Tuple[int, int],
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    # print(pixels)
    # Flip image horizontally
    pixels[:, 0] = viz_img_size[0] - pixels[:, 0]

    return pixels


def plot_trajs_and_points_on_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
    viz_img_size: Tuple[int, int],
    resize_factor:bool=False, 
):
    """
    Plot trajectories and points on an image.
    resize_factor: if True resize the image to viz_img_size. This is needed due to the fact that orginal image coming from fisheye is 640 x 480 and the traversability image is 224 x 224.
    Thus the camera matrix needs to be scaled accordingly.
    """
    # TODO this has to be in yaml config
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

        color = tuple(int(x) for x in np.random.choice(range(50, 255), size=3))

        # inverting x,y axis so origin in image is down-left corner
        if resize_factor:
            points[:, :, 1] = viz_img_size[1] * .46  - 1 - points[:, :, 1]
        else:
            points[:, :, 1] = viz_img_size[1] - 1 - points[:, :, 1]

        # Draw trajectory
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=2)

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

    # Color (red)
    marker.color.a = 1.0  # alpha
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    waypoint_viz_pub.publish(marker)


def make_marker_array(naction: np.ndarray, marker_array_pub):
    ma = MarkerArray()
    for idx, paths in enumerate(naction):
        r = 0.0
        g = 0.0
        b = 1.0
        marker = make_path_marker(
            paths, idx, r, g, b, frame_id="base_link")
        ma.markers.append(marker)
    marker_array_pub.publish(ma)


def publish_overlay_image(camera_matrix_orig, dist_coeffs, img: np.ndarray, pub: rospy.Publisher, trajs: List[np.ndarray], viz_img_size: Tuple[int, int], resize_factor:bool=False ):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Convert RGB → BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = plot_trajs_and_points_on_image(
        img=img,
        camera_matrix=camera_matrix_orig,
        dist_coeffs=dist_coeffs,
        list_trajs=trajs,
        viz_img_size=viz_img_size,
        resize_factor=resize_factor
    )

    ros_img = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    ros_img.header.stamp = rospy.Time.now()
    ros_img.header.frame_id = "base_footprint"
    pub.publish(ros_img)
